import hashlib
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import librosa
import mido
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


SR = 22050
HOP = 256
FMIN = librosa.note_to_hz("A0")
N_BINS = 88 * 2
BINS_PER_OCTAVE = 24
N_MELS = 176
PATCH_SIZE = 32
PRE_CONTEXT = 0.08
POST_CONTEXT = 0.12
FREQ_RADIUS_BINS = 24
MATCH_TOLERANCE = 0.10
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_midi_notes(midi_path: str) -> List[Dict]:
    mid = mido.MidiFile(midi_path)
    tempo = 500000
    tpq = mid.ticks_per_beat
    time_sec = 0.0
    active = {}
    notes = []
    for msg in mido.merge_tracks(mid.tracks):
        time_sec += mido.tick2second(msg.time, tpq, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append((time_sec, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            stack = active.get(msg.note)
            if stack:
                onset, vel = stack.pop()
                notes.append(
                    {
                        "onset": onset,
                        "offset": max(time_sec, onset + 0.01),
                        "pitch": int(msg.note),
                        "velocity": int(vel),
                    }
                )
    notes.sort(key=lambda x: (x["onset"], x["pitch"]))
    return notes


def compute_cache_key(wav_path: str) -> str:
    return hashlib.md5(wav_path.encode("utf-8")).hexdigest()[:16]


def load_audio_mono(wav_path: str) -> np.ndarray:
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def compute_cqt_cached(wav_path: str, cache_dir: str, overwrite: bool = False) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{compute_cache_key(wav_path)}_cqt.npy")
    if os.path.exists(cache_path) and not overwrite:
        return np.load(cache_path)

    audio = load_audio_mono(wav_path)
    cqt = librosa.cqt(
        y=audio,
        sr=SR,
        hop_length=HOP,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    mag = np.abs(cqt)
    db = librosa.amplitude_to_db(mag, ref=np.max)
    db = np.clip(db, -80.0, 0.0)
    norm = (db + 80.0) / 80.0
    np.save(cache_path, norm.astype(np.float32))
    return norm.astype(np.float32)


def compute_mel_cached(wav_path: str, cache_dir: str, overwrite: bool = False) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{compute_cache_key(wav_path)}_mel.npy")
    if os.path.exists(cache_path) and not overwrite:
        return np.load(cache_path)

    audio = load_audio_mono(wav_path)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=2048,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=SR / 2,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    db = np.clip(db, -80.0, 0.0)
    norm = (db + 80.0) / 80.0
    np.save(cache_path, norm.astype(np.float32))
    return norm.astype(np.float32)


def piece_name_from_wav(wav_path: str) -> str:
    piece = os.path.splitext(os.path.basename(wav_path))[0]
    if piece.endswith("_pcm"):
        piece = piece[:-4]
    return piece


def split_pieces(pkl_dir: str) -> Dict[str, str]:
    piece_to_split = {}
    non_test = []
    for pkl_path in sorted(os.listdir(pkl_dir)):
        if not pkl_path.endswith(".pkl"):
            continue
        with open(os.path.join(pkl_dir, pkl_path), "rb") as f:
            obj = pickle.load(f)
        wav_path = obj["wav_path"]
        piece = piece_name_from_wav(wav_path)
        if "ENSTDkCl_2" in wav_path or "ENSTDkAm_2" in wav_path:
            piece_to_split[piece] = "test"
        else:
            non_test.append(piece)
    non_test = sorted(set(non_test))
    rng = random.Random(SEED)
    rng.shuffle(non_test)
    n_train = int(round(len(non_test) * 0.85))
    train_set = set(non_test[:n_train])
    for piece in non_test:
        piece_to_split[piece] = "train" if piece in train_set else "val"
    return piece_to_split


def match_note(pred_note: Dict, gt_notes: Sequence[Dict], tolerance: float = MATCH_TOLERANCE) -> Optional[Dict]:
    best = None
    best_dt = tolerance
    for gt in gt_notes:
        if gt["pitch"] != pred_note["pitch"]:
            continue
        dt = abs(gt["onset"] - pred_note["onset"])
        if dt <= best_dt:
            best_dt = dt
            best = gt
    return best


def build_note_context(notes: Sequence[Dict], idx: int) -> np.ndarray:
    note = notes[idx]
    onset = float(note["onset"])
    offset = float(note["offset"])
    pitch = int(note["pitch"])
    velocity = float(note.get("velocity", 80))
    duration = max(offset - onset, 0.03)

    prev_gap = 1.0
    next_gap = 1.0
    prev_same_gap = 1.0
    next_same_gap = 1.0
    overlap_left = 0.0
    overlap_right = 0.0
    density_100 = 0.0
    density_500 = 0.0

    for j, other in enumerate(notes):
        if j == idx:
            continue
        dt = float(other["onset"]) - onset
        if abs(dt) <= 0.10:
            density_100 += 1.0
        if abs(dt) <= 0.50:
            density_500 += 1.0

    for j in range(idx - 1, -1, -1):
        other = notes[j]
        prev_gap = np.clip(onset - float(other["offset"]), -0.5, 1.0)
        overlap_left = max(0.0, float(other["offset"]) - onset)
        if int(other["pitch"]) == pitch:
            prev_same_gap = np.clip(onset - float(other["offset"]), -0.5, 1.0)
            break
        if j == idx - 1:
            prev_same_gap = 1.0
        if prev_gap != 1.0:
            break

    for j in range(idx + 1, len(notes)):
        other = notes[j]
        next_gap = np.clip(float(other["onset"]) - offset, -0.5, 1.0)
        overlap_right = max(0.0, offset - float(other["onset"]))
        if int(other["pitch"]) == pitch:
            next_same_gap = np.clip(float(other["onset"]) - offset, -0.5, 1.0)
            break
        if j == idx + 1:
            next_same_gap = 1.0
        if next_gap != 1.0:
            break

    context = np.array(
        [
            (pitch - 21.0) / 87.0,
            np.clip(duration / 2.0, 0.0, 1.0),
            np.clip(velocity / 127.0, 0.0, 1.0),
            np.clip((prev_gap + 0.5) / 1.5, 0.0, 1.0),
            np.clip((next_gap + 0.5) / 1.5, 0.0, 1.0),
            np.clip((prev_same_gap + 0.5) / 1.5, 0.0, 1.0),
            np.clip((next_same_gap + 0.5) / 1.5, 0.0, 1.0),
            np.clip(overlap_left / 0.5, 0.0, 1.0),
            np.clip(overlap_right / 0.5, 0.0, 1.0),
            np.clip(density_100 / 8.0, 0.0, 1.0),
            np.clip(density_500 / 24.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return context


def hz_to_bin(freq_hz: float) -> float:
    return BINS_PER_OCTAVE * np.log2(freq_hz / FMIN)


def midi_to_bin(midi_pitch: int) -> float:
    return hz_to_bin(librosa.midi_to_hz(midi_pitch))


def hz_to_mel_bin(freq_hz: float) -> float:
    mel_min = librosa.hz_to_mel(FMIN)
    mel_max = librosa.hz_to_mel(SR / 2)
    mel = librosa.hz_to_mel(freq_hz)
    return (mel - mel_min) / max(mel_max - mel_min, 1e-8) * (N_MELS - 1)


def safe_interval(start: float, end: float):
    if end <= start + 1e-3:
        end = start + 1e-3
    return start, end


def bilinear_resize(array: np.ndarray, size: int = PATCH_SIZE) -> np.ndarray:
    tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def crop_and_resize(feature_map: np.ndarray, onset: float, offset: float, pitch: int, feature_type: str = "cqt"):
    if feature_type == "mel":
        pitch_bin = hz_to_mel_bin(librosa.midi_to_hz(pitch))
    else:
        pitch_bin = midi_to_bin(pitch)
    time_start = max(0.0, onset - PRE_CONTEXT)
    time_end = max(time_start + 0.08, offset + POST_CONTEXT)
    frame_start = int(np.floor(time_start * SR / HOP))
    frame_end = int(np.ceil(time_end * SR / HOP))
    if frame_end <= frame_start + 2:
        frame_end = frame_start + 3

    f_center = int(round(pitch_bin))
    f0 = max(0, f_center - FREQ_RADIUS_BINS)
    f1 = min(feature_map.shape[0], f_center + FREQ_RADIUS_BINS)
    raw = np.zeros((2 * FREQ_RADIUS_BINS, max(3, frame_end - frame_start)), dtype=np.float32)
    src = feature_map[f0:f1, frame_start:frame_end]
    raw[: src.shape[0], : src.shape[1]] = src

    energy = raw.copy()
    delta = np.diff(raw, axis=1, prepend=raw[:, :1])
    delta = np.maximum(delta, 0.0)
    energy = bilinear_resize(energy)
    delta = bilinear_resize(delta)
    energy = np.clip(energy, 0.0, 1.0)
    delta = np.clip(delta / (delta.max() + 1e-6), 0.0, 1.0)

    patch_duration = time_end - time_start
    onset_n = np.clip((onset - time_start) / patch_duration, 0.0, 1.0)
    offset_n = np.clip((offset - time_start) / patch_duration, 0.0, 1.0)
    mu_t = np.clip((onset_n + offset_n) * 0.5, 0.0, 1.0)
    sigma_t = np.clip((offset_n - onset_n) * 0.5, 0.01, 0.45)
    mu_f = 0.5
    sigma_f = 0.10
    theta = 0.0
    amp = float(np.max(energy))

    meta = {
        "time_start": time_start,
        "time_end": time_end,
        "patch_duration": patch_duration,
        "pitch": int(pitch),
        "piece": None,
    }
    init_params = np.array([mu_t, mu_f, sigma_t, sigma_f, theta, amp], dtype=np.float32)
    patch = np.stack([energy, delta], axis=0).astype(np.float32)
    return patch, init_params, meta


@dataclass
class PieceRecord:
    piece: str
    wav_path: str
    midi_path: str
    pred_notes: List[Dict]
    gt_notes: List[Dict]


def load_piece_records(pkl_dir: str) -> Dict[str, List[PieceRecord]]:
    split_map = split_pieces(pkl_dir)
    grouped = {"train": [], "val": [], "test": []}
    for name in sorted(os.listdir(pkl_dir)):
        if not name.endswith(".pkl"):
            continue
        with open(os.path.join(pkl_dir, name), "rb") as f:
            obj = pickle.load(f)
        wav_path = obj["wav_path"].replace("\\", os.sep).replace("/", os.sep)
        midi_base = os.path.splitext(wav_path)[0]
        if midi_base.endswith("_pcm"):
            midi_base = midi_base[:-4]
        midi_path = midi_base + ".mid"
        piece = piece_name_from_wav(wav_path)
        if not os.path.exists(midi_path):
            continue
        grouped[split_map[piece]].append(
            PieceRecord(
                piece=piece,
                wav_path=wav_path,
                midi_path=midi_path,
                pred_notes=obj["notes"],
                gt_notes=load_midi_notes(midi_path),
            )
        )
    return grouped


class NoteRefinementDataset(Dataset):
    def __init__(
        self,
        records: Sequence[PieceRecord],
        cache_dir: str,
        split: str,
        overwrite_cache: bool = False,
        feature_type: str = "cqt",
    ):
        self.records = list(records)
        self.samples = []
        self.split = split
        self.feature_type = feature_type
        for record in self.records:
            sorted_notes = sorted(record.pred_notes, key=lambda x: (x["onset"], x["pitch"], x["offset"]))
            if feature_type == "mel":
                feature_map = compute_mel_cached(record.wav_path, cache_dir, overwrite=overwrite_cache)
            else:
                feature_map = compute_cqt_cached(record.wav_path, cache_dir, overwrite=overwrite_cache)
            for note_idx, pred_note in enumerate(sorted_notes):
                patch, init_params, meta = crop_and_resize(
                    feature_map,
                    pred_note["onset"],
                    pred_note["offset"],
                    pred_note["pitch"],
                    feature_type=feature_type,
                )
                meta["piece"] = record.piece
                meta["baseline_note"] = dict(pred_note)
                meta["feature_type"] = feature_type
                gt = match_note(pred_note, record.gt_notes)
                sample = {
                    "piece": record.piece,
                    "patch": patch,
                    "init_params": init_params,
                    "context": build_note_context(sorted_notes, note_idx),
                    "meta": meta,
                    "baseline_note": dict(pred_note),
                    "matched": gt is not None,
                }
                if gt is not None:
                    gt_on = np.clip((gt["onset"] - meta["time_start"]) / meta["patch_duration"], 0.0, 1.0)
                    gt_off = np.clip((gt["offset"] - meta["time_start"]) / meta["patch_duration"], 0.0, 1.0)
                    gt_mu_t = np.clip((gt_on + gt_off) * 0.5, 0.0, 1.0)
                    gt_sigma_t = np.clip((gt_off - gt_on) * 0.5, 0.01, 0.45)
                    gt_params = np.array(
                        [gt_mu_t, 0.5, gt_sigma_t, 0.10, 0.0, init_params[5]],
                        dtype=np.float32,
                    )
                    sample["gt_params"] = gt_params
                    sample["gt_note"] = gt
                self.samples.append(sample)
        if split != "test":
            self.samples = [s for s in self.samples if s["matched"]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        batch = {
            "patch": torch.from_numpy(sample["patch"]),
            "init_params": torch.from_numpy(sample["init_params"]),
            "context": torch.from_numpy(sample["context"]),
            "piece": sample["piece"],
            "baseline_note": sample["baseline_note"],
            "meta": sample["meta"],
            "matched": sample["matched"],
        }
        if sample["matched"]:
            batch["gt_params"] = torch.from_numpy(sample["gt_params"])
            batch["gt_note"] = sample["gt_note"]
        return batch
