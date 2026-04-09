import csv
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import mir_eval
import numpy as np
import torch
from torch.utils.data import DataLoader


def collate_notes(batch):
    result = {
        "patch": torch.stack([x["patch"] for x in batch], dim=0),
        "init_params": torch.stack([x["init_params"] for x in batch], dim=0),
        "context": torch.stack([x["context"] for x in batch], dim=0),
        "piece": [x["piece"] for x in batch],
        "baseline_note": [x["baseline_note"] for x in batch],
        "meta": [x["meta"] for x in batch],
        "matched": [x["matched"] for x in batch],
    }
    if all("gt_params" in x for x in batch):
        result["gt_params"] = torch.stack([x["gt_params"] for x in batch], dim=0)
        result["gt_note"] = [x["gt_note"] for x in batch]
    return result


def params_to_note(
    params: np.ndarray,
    meta: Dict,
    baseline_note: Dict,
    keep_baseline_onset: bool = True,
    offset_alpha: float = 1.0,
) -> Dict:
    time_start = meta["time_start"]
    patch_duration = meta["patch_duration"]
    mu_t, _, sigma_t, _, _, _ = params.tolist()
    onset = time_start + (mu_t - sigma_t) * patch_duration
    offset = time_start + (mu_t + sigma_t) * patch_duration
    onset = max(0.0, onset)
    offset = max(onset + 0.02, offset)

    base_onset = baseline_note["onset"]
    base_offset = baseline_note["offset"]
    dur = max(base_offset - base_onset, 0.03)

    onset = float(np.clip(onset, base_onset - 0.04, base_onset + 0.04))
    if keep_baseline_onset:
        onset = float(base_onset)

    offset = float(np.clip(offset, onset + 0.02, base_offset + max(0.20, 0.4 * dur)))
    offset = float((1.0 - offset_alpha) * base_offset + offset_alpha * offset)
    offset = max(offset, onset + 0.02)

    return {
        "onset": onset,
        "offset": offset,
        "pitch": int(baseline_note["pitch"]),
        "velocity": int(baseline_note.get("velocity", 80)),
    }


def notes_to_arrays(notes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    intervals = np.array([[n["onset"], n["offset"]] for n in notes], dtype=np.float64)
    pitches = mir_eval.util.midi_to_hz(np.array([n["pitch"] for n in notes], dtype=np.float64))
    return intervals, pitches


def evaluate_piece_metrics(pred_by_piece: Dict[str, List[Dict]], records) -> Tuple[List[Dict], Dict[str, float]]:
    rows = []
    for record in records:
        ref_iv, ref_hz = notes_to_arrays(record.gt_notes)
        est_iv, est_hz = notes_to_arrays(pred_by_piece[record.piece])

        note_p, note_r, note_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_iv,
            ref_hz,
            est_iv,
            est_hz,
            onset_tolerance=0.05,
            offset_ratio=0.2,
            offset_min_tolerance=0.05,
        )
        onset_p, onset_r, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_iv,
            ref_hz,
            est_iv,
            est_hz,
            onset_tolerance=0.05,
            offset_ratio=None,
        )
        rows.append(
            {
                "file": record.piece,
                "note_p": note_p,
                "note_r": note_r,
                "note_f1": note_f1,
                "onset_p": onset_p,
                "onset_r": onset_r,
                "onset_f1": onset_f1,
            }
        )

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "file"}
    return rows, summary


def build_predictions(
    model,
    dataset,
    device: str,
    use_model: bool = True,
    keep_baseline_onset: bool = True,
    offset_alpha: float = 1.0,
) -> Dict[str, List[Dict]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_notes)
    pred_by_piece = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            init_params = batch["init_params"].to(device)
            context = batch["context"].to(device)
            if use_model:
                patch = batch["patch"].to(device)
                refined, _ = model(patch, init_params, context=context)
                params_np = refined.cpu().numpy()
            else:
                params_np = init_params.cpu().numpy()
            for params, meta, base, piece in zip(params_np, batch["meta"], batch["baseline_note"], batch["piece"]):
                pred_by_piece[piece].append(
                    params_to_note(
                        params,
                        meta,
                        base,
                        keep_baseline_onset=keep_baseline_onset,
                        offset_alpha=offset_alpha,
                    )
                )
    return pred_by_piece


def compute_loss_terms(
    refined_params,
    gt_params,
    renderer,
    patch,
    loss_variant: str = "base",
    use_reconstruction_loss: bool = True,
):
    pred_onset = refined_params[:, 0] - refined_params[:, 2]
    pred_offset = refined_params[:, 0] + refined_params[:, 2]
    gt_onset = gt_params[:, 0] - gt_params[:, 2]
    gt_offset = gt_params[:, 0] + gt_params[:, 2]

    loss_onset = torch.nn.functional.smooth_l1_loss(pred_onset, gt_onset)
    loss_offset = torch.nn.functional.smooth_l1_loss(pred_offset, gt_offset)
    loss_mu = torch.nn.functional.smooth_l1_loss(refined_params[:, 0], gt_params[:, 0])
    loss_sigma = torch.nn.functional.smooth_l1_loss(refined_params[:, 2], gt_params[:, 2])
    rendered = renderer(refined_params)
    recon_target = patch[:, 0]
    if loss_variant == "contour":
        if use_reconstruction_loss:
            time_contour = patch[:, 1].clamp(0.0, 1.0)
            freq_grad = torch.abs(recon_target[:, 1:] - recon_target[:, :-1])
            freq_grad = torch.nn.functional.pad(freq_grad, (0, 0, 0, 1))
            weight = 1.0 + 2.0 * time_contour + 1.0 * freq_grad
            loss_recon = (((rendered - recon_target) ** 2) * weight).mean()
            loss_tail = torch.nn.functional.smooth_l1_loss(
                (rendered * weight).sum(dim=1),
                (recon_target * weight).sum(dim=1),
            )
        else:
            loss_recon = torch.zeros((), device=rendered.device)
            loss_tail = torch.zeros((), device=rendered.device)
        total = (
            2.0 * loss_onset
            + 1.75 * loss_offset
            + 0.5 * loss_mu
            + 0.75 * loss_sigma
            + (0.15 * loss_recon if use_reconstruction_loss else 0.0)
            + (0.05 * loss_tail if use_reconstruction_loss else 0.0)
        )
    else:
        if use_reconstruction_loss:
            loss_recon = torch.nn.functional.mse_loss(rendered, recon_target)
        else:
            loss_recon = torch.zeros((), device=rendered.device)
        loss_tail = torch.zeros((), device=rendered.device)
        total = (
            2.0 * loss_onset
            + 1.5 * loss_offset
            + 0.5 * loss_mu
            + 0.5 * loss_sigma
            + (0.1 * loss_recon if use_reconstruction_loss else 0.0)
        )
    return total, {
        "loss_total": float(total.item()),
        "loss_onset": float(loss_onset.item()),
        "loss_offset": float(loss_offset.item()),
        "loss_recon": float(loss_recon.item()),
        "loss_tail": float(loss_tail.item()),
    }


def evaluate_validation(model, dataset, renderer, device: str):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_notes)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            patch = batch["patch"].to(device)
            init_params = batch["init_params"].to(device)
            context = batch["context"].to(device)
            gt_params = batch["gt_params"].to(device)
            refined, _ = model(patch, init_params, context=context)
            loss, _ = compute_loss_terms(refined, gt_params, renderer, patch)
            losses.append(float(loss.item()))
    pred_by_piece = build_predictions(model, dataset, device, use_model=True)
    baseline_by_piece = build_predictions(model, dataset, device, use_model=False)
    refined_rows, refined_summary = evaluate_piece_metrics(pred_by_piece, dataset.records)
    _, baseline_summary = evaluate_piece_metrics(baseline_by_piece, dataset.records)
    refined_summary["val_loss"] = float(np.mean(losses)) if losses else math.nan
    return baseline_summary, refined_summary, refined_rows


def save_piece_predictions(pred_by_piece: Dict[str, List[Dict]], records, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    merged = {}
    for record in records:
        payload = {"wav_path": record.wav_path, "notes": pred_by_piece[record.piece]}
        merged[record.piece] = payload
        with open(os.path.join(output_dir, f"{record.piece}.pkl"), "wb") as f:
            pickle.dump(payload, f)
    with open(os.path.join(output_dir, "refined_predictions.pkl"), "wb") as f:
        pickle.dump(merged, f)


def save_results_table(rows: List[Dict], summary: Dict[str, float], output_csv: str) -> None:
    fieldnames = ["file", "note_p", "note_r", "note_f1", "onset_p", "onset_r", "onset_f1"]
    summary_row = {"file": "MEAN"}
    for key in fieldnames[1:]:
        if key in summary:
            summary_row[key] = summary[key]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(summary_row)


def save_ablation_json(output_json: str, ablation: str, summary: Dict[str, float]) -> None:
    payload = {
        "ablation": ablation,
        "note_precision": float(summary["note_p"]),
        "note_recall": float(summary["note_r"]),
        "note_f1": float(summary["note_f1"]),
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
