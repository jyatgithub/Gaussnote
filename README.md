# GaussNote

This repository hosts the public release of the Gaussian note refinement module used in our automatic piano transcription experiments.

## Included component

- `gaussnote_refine/`: proposal-conditioned Gaussian ellipsoid refinement for piano AMT

The released module refines coarse note proposals using:

- local CQT or mel patches
- a differentiable Gaussian ellipsoid renderer
- a lightweight CNN refinement network
- an optional gated adapter for context-aware parameter updates

## Repository layout

```text
Gaussnote/
  gaussnote_refine/
    dataset.py
    model.py
    renderer.py
    evaluate.py
    train.py
    requirements.txt
```

## Quick start

Install dependencies:

```bash
pip install -r gaussnote_refine/requirements.txt
```

Train or evaluate the refinement module on a directory of coarse AMT note proposals:

```bash
python -m gaussnote_refine.train --pkl-dir ./yourdir/yourfile --output-dir ./yourdir/yourfile
```

For more details, see [gaussnote_refine/README.md](./gaussnote_refine/README.md).
