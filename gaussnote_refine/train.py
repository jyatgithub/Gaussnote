import argparse
import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from gaussnote_refine.dataset import (
    NoteRefinementDataset,
    load_piece_records,
    seed_everything,
)
from gaussnote_refine.evaluate import (
    build_predictions,
    collate_notes,
    compute_loss_terms,
    evaluate_piece_metrics,
    save_ablation_json,
    save_piece_predictions,
    save_results_table,
)
from gaussnote_refine.model import RefinementNet, count_parameters
from gaussnote_refine.renderer import GaussianEllipsoidRenderer


def run_epoch(
    model,
    loader,
    optimizer,
    renderer,
    device,
    loss_variant: str = "base",
    use_reconstruction_loss: bool = True,
):
    model.train()
    losses = []
    for batch in loader:
        patch = batch["patch"].to(device)
        init_params = batch["init_params"].to(device)
        context = batch["context"].to(device)
        gt_params = batch["gt_params"].to(device)

        refined, _ = model(patch, init_params, context=context)
        loss, _ = compute_loss_terms(
            refined,
            gt_params,
            renderer,
            patch,
            loss_variant=loss_variant,
            use_reconstruction_loss=use_reconstruction_loss,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_split(
    model,
    dataset,
    renderer,
    device,
    use_model: bool,
    keep_baseline_onset: bool = True,
    offset_alpha: float = 1.0,
    loss_variant: str = "base",
    use_reconstruction_loss: bool = True,
):
    has_targets = len(dataset) > 0 and dataset.split != "test"
    if use_model and has_targets:
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
                loss, _ = compute_loss_terms(
                    refined,
                    gt_params,
                    renderer,
                    patch,
                    loss_variant=loss_variant,
                    use_reconstruction_loss=use_reconstruction_loss,
                )
                losses.append(float(loss.item()))
        val_loss = float(np.mean(losses)) if losses else 0.0
    else:
        val_loss = 0.0

    pred_by_piece = build_predictions(
        model,
        dataset,
        device,
        use_model=use_model,
        keep_baseline_onset=keep_baseline_onset,
        offset_alpha=offset_alpha,
    )
    rows, summary = evaluate_piece_metrics(pred_by_piece, dataset.records)
    summary["loss"] = val_loss
    return rows, summary, pred_by_piece


def print_epoch(epoch, train_loss, val_summary, lr, elapsed):
    print(
        f"{epoch:03d} | "
        f"{train_loss:.5f} | "
        f"{val_summary['loss']:.5f} | "
        f"{val_summary['onset_f1']:.4f} | "
        f"{val_summary['note_f1']:.4f} | "
        f"{lr:.6f} | "
        f"{elapsed/60.0:.1f}m"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-dir", default=os.environ.get("PKL_DIR", "./pkl_results"))
    parser.add_argument("--cache-dir", default="./refine_cache")
    parser.add_argument("--output-dir", default="./gaussnote_refine_outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--time-limit-min", type=float, default=120.0)
    parser.add_argument("--feature-type", choices=["cqt", "mel"], default="cqt")
    parser.add_argument("--model-variant", choices=["base", "adapter"], default="base")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--loss-variant", choices=["base", "contour"], default="base")
    parser.add_argument("--ablation", choices=["none", "A1", "A2", "A3", "A4", "A5"], default="none")
    parser.add_argument("--drop-patch-embedding", action="store_true")
    parser.add_argument("--drop-ellipsoid-embedding", action="store_true")
    parser.add_argument("--drop-context-embedding", action="store_true")
    parser.add_argument("--disable-param-gate", action="store_true")
    parser.add_argument("--disable-reconstruction-loss", action="store_true")
    parser.add_argument("--disable-baseline-fallback", action="store_true")
    parser.add_argument("--result-tag", default="")
    args = parser.parse_args()

    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    records = load_piece_records(args.pkl_dir)
    train_set = NoteRefinementDataset(
        records["train"],
        args.cache_dir,
        "train",
        overwrite_cache=args.overwrite_cache,
        feature_type=args.feature_type,
    )
    val_set = NoteRefinementDataset(
        records["val"],
        args.cache_dir,
        "val",
        overwrite_cache=args.overwrite_cache,
        feature_type=args.feature_type,
    )
    test_set = NoteRefinementDataset(
        records["test"],
        args.cache_dir,
        "test",
        overwrite_cache=False,
        feature_type=args.feature_type,
    )
    train_set.records = records["train"]
    val_set.records = records["val"]
    test_set.records = records["test"]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_notes)

    context_dim = 11
    model = RefinementNet(
        context_dim=context_dim,
        use_adapter=(args.model_variant == "adapter"),
        ablation=args.ablation,
        drop_patch_embedding=args.drop_patch_embedding,
        drop_ellipsoid_embedding=args.drop_ellipsoid_embedding,
        drop_context_embedding=args.drop_context_embedding,
        disable_param_gate=args.disable_param_gate,
    ).to(device)
    if args.freeze_backbone:
        model.freeze_backbone()
    renderer = GaussianEllipsoidRenderer().to(device)
    use_reconstruction_loss = args.ablation != "A4" and not args.disable_reconstruction_loss
    result_tag = args.result_tag or (args.ablation if args.ablation != "none" else "full")
    n_params = count_parameters(model)
    print(f"device={device}")
    print(f"trainable_params={n_params}")
    print(f"feature_type={args.feature_type}")
    print(f"model_variant={args.model_variant}")
    print(f"freeze_backbone={args.freeze_backbone}")
    print(f"loss_variant={args.loss_variant}")
    print(f"ablation={args.ablation}")
    print(f"drop_patch_embedding={args.drop_patch_embedding}")
    print(f"drop_ellipsoid_embedding={args.drop_ellipsoid_embedding}")
    print(f"drop_context_embedding={args.drop_context_embedding}")
    print(f"disable_param_gate={args.disable_param_gate}")
    print(f"use_reconstruction_loss={use_reconstruction_loss}")
    print(f"disable_baseline_fallback={args.disable_baseline_fallback}")
    print(f"result_tag={result_tag}")
    if n_params >= 2_000_000:
        raise RuntimeError("Parameter budget exceeded")

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    baseline_val_rows, baseline_val_summary, _ = evaluate_split(model, val_set, renderer, device, use_model=False)
    baseline_test_rows, baseline_test_summary, baseline_test_preds = evaluate_split(model, test_set, renderer, device, use_model=False)
    print(
        f"baseline_val onset_f1={baseline_val_summary['onset_f1']:.4f} "
        f"note_f1={baseline_val_summary['note_f1']:.4f}"
    )
    print(
        f"baseline_test onset_f1={baseline_test_summary['onset_f1']:.4f} "
        f"note_f1={baseline_test_summary['note_f1']:.4f}"
    )

    best_state = None
    best_summary = None
    best_epoch = 0
    best_val_note_f1 = baseline_val_summary["note_f1"]
    no_improve = 0
    patience = 5
    start_time = time.time()

    print("epoch | train_loss | val_loss | val_onset_f1 | val_note_f1 | lr | elapsed")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            renderer,
            device,
            loss_variant=args.loss_variant,
            use_reconstruction_loss=use_reconstruction_loss,
        )
        val_rows, val_summary, _ = evaluate_split(
            model,
            val_set,
            renderer,
            device,
            use_model=True,
            keep_baseline_onset=True,
            offset_alpha=1.0,
            loss_variant=args.loss_variant,
            use_reconstruction_loss=use_reconstruction_loss,
        )
        scheduler.step(val_summary["note_f1"])
        elapsed = time.time() - start_time
        print_epoch(epoch, train_loss, val_summary, optimizer.param_groups[0]["lr"], elapsed)

        if val_summary["note_f1"] > best_val_note_f1 + 0.005:
            best_val_note_f1 = val_summary["note_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_summary = val_summary
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break
        if elapsed > max(60.0, (args.time_limit_min - 5.0) * 60.0):
            break

    use_refined = best_state is not None or args.disable_baseline_fallback
    if use_refined:
        if best_state is not None:
            model.load_state_dict(best_state)

    refined_val_rows, refined_val_summary, _ = evaluate_split(
        model,
        val_set,
        renderer,
        device,
        use_model=use_refined,
        keep_baseline_onset=True,
        offset_alpha=1.0,
        loss_variant=args.loss_variant,
        use_reconstruction_loss=use_reconstruction_loss,
    )
    if (
        not args.disable_baseline_fallback
        and (
            refined_val_summary["note_f1"] < baseline_val_summary["note_f1"]
            or refined_val_summary["onset_f1"] < baseline_val_summary["onset_f1"] - 0.01
        )
    ):
        use_refined = False

    final_test_rows, final_test_summary, final_test_preds = evaluate_split(
        model,
        test_set,
        renderer,
        device,
        use_model=use_refined,
        keep_baseline_onset=True,
        offset_alpha=1.0,
        loss_variant=args.loss_variant,
        use_reconstruction_loss=use_reconstruction_loss,
    )
    if not use_refined:
        final_test_rows = baseline_test_rows
        final_test_summary = baseline_test_summary
        final_test_preds = baseline_test_preds

    results_table = [
        {
            "metric": "onset F1",
            "baseline": baseline_test_summary["onset_f1"],
            "refined": final_test_summary["onset_f1"],
            "delta": final_test_summary["onset_f1"] - baseline_test_summary["onset_f1"],
        },
        {
            "metric": "note F1",
            "baseline": baseline_test_summary["note_f1"],
            "refined": final_test_summary["note_f1"],
            "delta": final_test_summary["note_f1"] - baseline_test_summary["note_f1"],
        },
        {
            "metric": "note precision",
            "baseline": baseline_test_summary["note_p"],
            "refined": final_test_summary["note_p"],
            "delta": final_test_summary["note_p"] - baseline_test_summary["note_p"],
        },
        {
            "metric": "note recall",
            "baseline": baseline_test_summary["note_r"],
            "refined": final_test_summary["note_r"],
            "delta": final_test_summary["note_r"] - baseline_test_summary["note_r"],
        },
    ]
    for row in results_table:
        print(
            f"{row['metric']:14s} | "
            f"{row['baseline']:.4f} | "
            f"{row['refined']:.4f} | "
            f"{row['delta']:+.4f}"
        )

    ckpt_path = os.path.join(args.output_dir, "gaussnote_refined.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "use_refined": use_refined,
            "best_epoch": best_epoch,
            "best_val_summary": best_summary,
            "baseline_val_summary": baseline_val_summary,
            "final_test_summary": final_test_summary,
            "feature_type": args.feature_type,
            "model_variant": args.model_variant,
            "freeze_backbone": args.freeze_backbone,
            "loss_variant": args.loss_variant,
            "ablation": args.ablation,
            "drop_patch_embedding": args.drop_patch_embedding,
            "drop_ellipsoid_embedding": args.drop_ellipsoid_embedding,
            "drop_context_embedding": args.drop_context_embedding,
            "disable_param_gate": args.disable_param_gate,
            "use_reconstruction_loss": use_reconstruction_loss,
            "disable_baseline_fallback": args.disable_baseline_fallback,
            "result_tag": result_tag,
        },
        ckpt_path,
    )
    pred_dir = os.path.join(args.output_dir, "refined_predictions")
    save_piece_predictions(final_test_preds, records["test"], pred_dir)
    save_results_table(final_test_rows, final_test_summary, os.path.join(args.output_dir, "results_evaall_test.csv"))
    if args.ablation != "none" or args.result_tag:
        save_ablation_json(
            os.path.join(args.output_dir, f"ablation_{result_tag}.json"),
            result_tag,
            final_test_summary,
        )

    print("Training/evaluation finished.")
    print(f"  final note_f1={final_test_summary['note_f1']:.4f}")
    print(f"  final onset_f1={final_test_summary['onset_f1']:.4f}")


if __name__ == "__main__":
    main()
