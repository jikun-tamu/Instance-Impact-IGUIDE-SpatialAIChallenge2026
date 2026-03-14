#!/usr/bin/env python3
"""Standalone Stage-2 ensemble inference with uncertainty metrics.

Features:
- 3-model weighted ensemble on cumulative CORAL logits (default weights: 4,3,2)
- Per-sample logging of each model logits/probs and ensemble logits/probs
- Uncertainty outputs:
  - weighted variance of p(model, ensemble_class)
  - weighted variance of expected severity E[y]
  - pmax, margin, entropy from ensemble probabilities
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_stage2 import (
    SiameseDamageModel,
    collate_batch,
    coral_probs_from_logits,
    load_mask_tensor,
    load_rgb_tensor,
    macro_f1_from_cm,
    confusion_matrix,
    qwk_from_cm,
    read_rows,
)


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble inference for Stage-2 checkpoints.")
    p.add_argument("--csv", type=Path, required=True, help="Path to stage2_samples.csv.")
    p.add_argument(
        "--ckpts",
        type=str,
        required=True,
        help="Comma-separated list of 3 checkpoint paths (stage2_best.pt files).",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="4,3,2",
        help="Comma-separated nonnegative ensemble weights aligned with --ckpts.",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="",
        help="Optional comma-separated train_config.json paths aligned with --ckpts. "
        "Use this when checkpoints are copied to a different directory.",
    )
    p.add_argument(
        "--calibration_dirs",
        type=str,
        default="",
        help="Optional comma-separated calibration output dirs aligned with --ckpts "
        "(each containing calibration_metrics.json).",
    )
    p.add_argument(
        "--calibration_method",
        choices=["none", "temperature", "vector"],
        default="none",
        help="Calibration method applied per-model before confidence metrics.",
    )
    p.add_argument("--out_jsonl", type=Path, required=True, help="Output JSONL path.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=0, help="Optional row limit for quick runs.")
    p.add_argument("--print_examples", type=int, default=5, help="How many sample rows to print in terminal.")
    p.add_argument("--log_every_steps", type=int, default=50)
    return p.parse_args()


class Stage2InferenceDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        pre = load_rgb_tensor(row["pre_crop"])
        post = load_rgb_tensor(row["post_crop"])
        m = load_mask_tensor(row["mask_M"])
        r = load_mask_tensor(row["mask_R"])
        y = int(row["damage_class"]) if "damage_class" in row and row["damage_class"] != "" else -1
        meta = {
            "tile_id": row.get("tile_id", ""),
            "event_id": row.get("event_id", ""),
            "bldg_uid": row.get("bldg_uid", ""),
        }
        return pre, post, m, r, y, meta, i


def parse_csv_list(raw, cast=float):
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return [cast(x) for x in parts]


def weighted_variance(values, weights):
    # values: [B, M], weights: [M]
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    mean = np.sum(values * w.reshape(1, -1), axis=1)
    var = np.sum(((values - mean.reshape(-1, 1)) ** 2) * w.reshape(1, -1), axis=1)
    return mean, var


def load_model_from_ckpt(ckpt_path, device, cfg_override=None):
    ckpt_path = Path(ckpt_path)
    cfg_path = Path(cfg_override) if cfg_override else (ckpt_path.parent / "train_config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing train_config.json next to checkpoint: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    model = SiameseDamageModel(
        backbone_name=cfg.get("backbone", "convnext_tiny"),
        num_classes=int(cfg.get("classes", 4)),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        dropout=float(cfg.get("dropout", 0.1)),
        pretrained=False,  # do not trigger downloads; checkpoint weights are loaded below
        stage_index=int(cfg.get("backbone_stage_index", -1)),
        change_fusion=cfg.get("change_fusion", "pre_post_diff"),
        diff_abs_scale=float(cfg.get("diff_abs_scale", 1.0)),
        pooling_mode=cfg.get("pooling_mode", "mask_m_ring"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def safe_round_list(xs, nd=4):
    return [round(float(x), nd) for x in xs]


def load_calibration_spec(calib_dir):
    metrics_path = Path(calib_dir) / "calibration_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing calibration metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    t = float(metrics["temperature"]["temperature"])
    exponents = [float(x) for x in metrics["vector_exponents"]["exponents"]]
    return {"temperature": t, "vector_exponents": exponents}


def apply_calibration_np(probs, logits_cum, method, spec):
    if method == "none" or spec is None:
        return probs
    if method == "temperature":
        # For CORAL, temperature scaling on cumulative logits then mapping back to class probs.
        t = max(1e-6, float(spec["temperature"]))
        logits_t = torch.from_numpy((logits_cum / t).astype(np.float32))
        p = coral_probs_from_logits(logits_t).cpu().numpy()
        return p
    # vector exponents on class probabilities.
    exps = np.asarray(spec["vector_exponents"], dtype=np.float64).reshape(1, -1)
    p = np.power(np.clip(probs, 1e-12, 1.0), exps)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return p


def main():
    args = parse_args()
    ckpts = [Path(x) for x in parse_csv_list(args.ckpts, cast=str)]
    cfg_overrides = parse_csv_list(args.configs, cast=str) if args.configs else []
    calib_dirs = parse_csv_list(args.calibration_dirs, cast=str) if args.calibration_dirs else []
    weights = parse_csv_list(args.weights, cast=float)
    if len(ckpts) != 3:
        raise ValueError(f"Expected exactly 3 checkpoints, got {len(ckpts)}")
    if len(weights) != len(ckpts):
        raise ValueError(f"weights length ({len(weights)}) must match checkpoints ({len(ckpts)})")
    if cfg_overrides and len(cfg_overrides) != len(ckpts):
        raise ValueError(f"configs length ({len(cfg_overrides)}) must match checkpoints ({len(ckpts)})")
    if calib_dirs and len(calib_dirs) != len(ckpts):
        raise ValueError(f"calibration_dirs length ({len(calib_dirs)}) must match checkpoints ({len(ckpts)})")
    if args.calibration_method != "none" and not calib_dirs:
        raise ValueError("calibration_method is set but no calibration_dirs were provided.")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights must be nonnegative and not all zero")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv, limit=args.limit)
    if len(rows) == 0:
        raise RuntimeError("No rows loaded from CSV.")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print("Loaded rows:", len(rows))
    print("Device:", device)
    print("Weights:", weights)
    print("Checkpoints:")
    for p in ckpts:
        print(" -", p)
    if cfg_overrides:
        print("Configs:")
        for c in cfg_overrides:
            print(" -", c)
    print("Calibration method:", args.calibration_method)
    calib_specs = []
    if calib_dirs:
        print("Calibration dirs:")
        for d in calib_dirs:
            print(" -", d)
            calib_specs.append(load_calibration_spec(d))
    else:
        calib_specs = [None] * len(ckpts)

    models = []
    cfgs = []
    for i, p in enumerate(ckpts):
        cfg_override = cfg_overrides[i] if cfg_overrides else None
        m, cfg = load_model_from_ckpt(p, device, cfg_override=cfg_override)
        models.append(m)
        cfgs.append(cfg)

    num_classes = int(cfgs[0].get("classes", 4))
    for c in cfgs[1:]:
        if int(c.get("classes", 4)) != num_classes:
            raise ValueError("All checkpoints must share same class count.")

    dataset = Stage2InferenceDataset(rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / np.sum(weights_arr)

    out_f = args.out_jsonl.open("w", encoding="utf-8")
    examples_printed = 0
    all_true = []
    all_pred = []
    pmax_list = []
    margin_list = []
    ent_list = []
    var_cls_list = []
    var_exp_list = []

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            pre, post, m, r, y, meta, _ = batch
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)

            model_logits = []
            model_probs = []
            model_probs_cal = []
            for model in models:
                out = model(pre, post, m, r)
                lg = out["logits_cum"].detach().float().cpu().numpy()  # [B, K-1]
                pb = out["probs"].detach().float().cpu().numpy()  # [B, K]
                model_logits.append(lg)
                model_probs.append(pb)
            for i_m in range(len(models)):
                p_cal = apply_calibration_np(
                    probs=model_probs[i_m],
                    logits_cum=model_logits[i_m],
                    method=args.calibration_method,
                    spec=calib_specs[i_m] if i_m < len(calib_specs) else None,
                )
                model_probs_cal.append(p_cal)

            # Weighted average on cumulative logits.
            ens_logits = np.zeros_like(model_logits[0], dtype=np.float64)
            for w_i, lg in zip(weights_arr, model_logits):
                ens_logits += w_i * lg

            ens_logits_t = torch.from_numpy(ens_logits.astype(np.float32))
            ens_probs = coral_probs_from_logits(ens_logits_t).cpu().numpy()
            ens_pred = ens_probs.argmax(axis=1)
            ens_probs_cal = np.zeros_like(model_probs_cal[0], dtype=np.float64)
            for w_i, p_i in zip(weights_arr, model_probs_cal):
                ens_probs_cal += w_i * p_i
            ens_probs_cal = ens_probs_cal / np.clip(ens_probs_cal.sum(axis=1, keepdims=True), 1e-12, None)
            ens_exp = (ens_probs_cal * np.arange(num_classes, dtype=np.float64).reshape(1, -1)).sum(axis=1)

            # Uncertainty 1: weighted variance of each model probability for ensemble-predicted class.
            p_cls_matrix = []
            exp_matrix = []
            for pb in model_probs:
                p_cls_matrix.append(pb[np.arange(pb.shape[0]), ens_pred])
                exp_i = (pb * np.arange(num_classes, dtype=np.float64).reshape(1, -1)).sum(axis=1)
                exp_matrix.append(exp_i)
            p_cls_matrix = np.stack(p_cls_matrix, axis=1)  # [B, 3]
            exp_matrix = np.stack(exp_matrix, axis=1)  # [B, 3]

            _, var_pred_class = weighted_variance(p_cls_matrix, weights_arr)
            _, var_expected = weighted_variance(exp_matrix, weights_arr)

            # Confidence on ensemble probs.
            pmax = np.max(ens_probs_cal, axis=1)
            sorted_probs = np.sort(ens_probs_cal, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            entropy = -np.sum(ens_probs_cal * np.log(np.clip(ens_probs_cal, 1e-12, 1.0)), axis=1)

            y_np = y.numpy()
            all_true.extend([int(v) for v in y_np.tolist()])
            all_pred.extend([int(v) for v in ens_pred.tolist()])
            pmax_list.extend(pmax.tolist())
            margin_list.extend(margin.tolist())
            ent_list.extend(entropy.tolist())
            var_cls_list.extend(var_pred_class.tolist())
            var_exp_list.extend(var_expected.tolist())

            bsz = ens_probs.shape[0]
            for i in range(bsz):
                rec = {
                    "sample_index": int((step - 1) * args.batch_size + i),
                    "tile_id": meta[i].get("tile_id", ""),
                    "event_id": meta[i].get("event_id", ""),
                    "bldg_uid": meta[i].get("bldg_uid", ""),
                    "y_true": int(y_np[i]),
                    "y_pred_ensemble": int(ens_pred[i]),
                    "expected_severity_ensemble": float(ens_exp[i]),
                    "var_predicted_class_prob_weighted": float(var_pred_class[i]),
                    "var_expected_severity_weighted": float(var_expected[i]),
                    "pmax": float(pmax[i]),
                    "margin": float(margin[i]),
                    "entropy": float(entropy[i]),
                    "weights": [float(x) for x in weights_arr.tolist()],
                    "model_logits_cum": [model_logits[m_i][i].astype(float).tolist() for m_i in range(3)],
                    "model_probs": [model_probs[m_i][i].astype(float).tolist() for m_i in range(3)],
                    "model_probs_calibrated": [model_probs_cal[m_i][i].astype(float).tolist() for m_i in range(3)],
                    "calibration_method": args.calibration_method,
                    "ensemble_logits_cum": ens_logits[i].astype(float).tolist(),
                    "ensemble_probs": ens_probs[i].astype(float).tolist(),
                    "ensemble_probs_calibrated": ens_probs_cal[i].astype(float).tolist(),
                }
                out_f.write(json.dumps(rec) + "\n")

                if examples_printed < args.print_examples:
                    print(
                        f"[example {examples_printed+1}] event={rec['event_id']} tile={rec['tile_id']} "
                        f"y_true={rec['y_true']} y_pred={rec['y_pred_ensemble']}"
                    )
                    for m_i in range(3):
                        print(
                            f"  model{m_i+1} logits_cum={safe_round_list(rec['model_logits_cum'][m_i])} "
                            f"probs={safe_round_list(rec['model_probs'][m_i])} "
                            f"probs_cal={safe_round_list(rec['model_probs_calibrated'][m_i])}"
                        )
                    print(
                        f"  ensemble logits_cum={safe_round_list(rec['ensemble_logits_cum'])} "
                        f"probs={safe_round_list(rec['ensemble_probs'])} "
                        f"probs_cal={safe_round_list(rec['ensemble_probs_calibrated'])}"
                    )
                    print(
                        f"  unc var_cls={rec['var_predicted_class_prob_weighted']:.6f} "
                        f"var_Ey={rec['var_expected_severity_weighted']:.6f} "
                        f"pmax={rec['pmax']:.4f} margin={rec['margin']:.4f} entropy={rec['entropy']:.4f}"
                    )
                    examples_printed += 1

            if args.log_every_steps > 0 and step % args.log_every_steps == 0:
                print(f"processed steps={step} rows={step * args.batch_size}")

    out_f.close()

    # If labels are present in CSV, compute quick metrics.
    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    labeled_mask = y_true >= 0
    if np.any(labeled_mask):
        cm = confusion_matrix(y_true[labeled_mask], y_pred[labeled_mask], n_classes=num_classes)
        macro_f1, per_class_f1 = macro_f1_from_cm(cm)
        qwk = qwk_from_cm(cm)
        print("Ensemble validation-like metrics:")
        print("  macro_f1:", f"{macro_f1:.4f}")
        print("  per_class_f1:", [round(x, 4) for x in per_class_f1])
        print("  qwk:", f"{qwk:.4f}")
    print("Uncertainty summary:")
    print("  mean var_predicted_class_prob_weighted:", f"{float(np.mean(var_cls_list)):.6f}")
    print("  mean var_expected_severity_weighted:", f"{float(np.mean(var_exp_list)):.6f}")
    print("  mean pmax/margin/entropy:", f"{float(np.mean(pmax_list)):.4f}", f"{float(np.mean(margin_list)):.4f}", f"{float(np.mean(ent_list)):.4f}")
    print("Wrote predictions:", args.out_jsonl)


if __name__ == "__main__":
    main()
