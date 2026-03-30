# gaussnote_refine

`gaussnote_refine` is a lightweight note refinement module for automatic piano transcription.
It takes coarse note proposals from an upstream AMT system and refines them with a Gaussian ellipsoid representation on the time-frequency plane.

## Method summary

Each note proposal is converted into an initial Gaussian parameter vector:

- `mu_t`: temporal center
- `mu_f`: frequency center
- `sigma_t`: temporal spread
- `sigma_f`: spectral spread
- `theta`: rotation
- `A`: amplitude

The model then uses:

1. a local acoustic patch (`CQT` or `mel`)
2. the initial Gaussian parameters
3. note-level context features

to predict conservative residual updates for the Gaussian parameters.

The released implementation supports:

- a base refinement network
- an adapter-enhanced variant with parameter-wise gating

## Files

- `dataset.py`: proposal loading, note matching, cached feature extraction, patch construction
- `model.py`: CNN backbone, refinement head, optional gated adapter
- `renderer.py`: differentiable 2D Gaussian ellipsoid renderer
- `evaluate.py`: note decoding, loss terms, mir_eval-based scoring
- `train.py`: training and evaluation entry point
- `requirements.txt`: Python dependencies

## Input format

The training script expects a directory of `.pkl` files, one file per piece.
Each pickle file should contain:

```python
{
    "wav_path": "...",
    "notes": [
        {"onset": float, "offset": float, "pitch": int, "velocity": int},
        ...
    ]
}
```

The script matches these coarse note proposals against note annotations derived from the paired MIDI file.

## Usage

Basic run:

```bash
python -m gaussnote_refine.train \
  --pkl-dir ./bytedance_preds \
  --cache-dir ./refine_cache \
  --output-dir ./gaussnote_refine_outputs
```

Adapter-based variant:

```bash
python -m gaussnote_refine.train \
  --pkl-dir ./bytedance_preds \
  --cache-dir ./refine_cache \
  --output-dir ./gaussnote_refine_outputs \
  --model-variant adapter
```

Useful options:

- `--feature-type cqt|mel`
- `--model-variant base|adapter`
- `--freeze-backbone`
- `--loss-variant base|contour`
- `--time-limit-min 120`

## Output

The script writes:

- `results_evaall_test.csv`: per-piece and mean note/onset metrics
- `gaussnote_refined.pt`: trained checkpoint
- `refined_predictions/`: per-piece refined note predictions in pickle format

## Notes

- The module is designed as a refinement stage rather than an end-to-end AMT system.
- It assumes coarse note proposals are already available.
- Cached acoustic features are stored under the specified cache directory to speed up reruns.
