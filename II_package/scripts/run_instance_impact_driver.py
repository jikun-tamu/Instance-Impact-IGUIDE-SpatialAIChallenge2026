#!/usr/bin/env python3
"""Run end-to-end Instance Impact driver on one pre/post image pair.

Pipeline:
1) Stage-1 SAM3 inference on one pre-disaster image
2) Shared instance crop/mask generation
3) Stage-2a inference
4) Stage-2b ensemble inference
5) Presentation merge/report
6) Instance-level visualization overlays
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end driver for one-image-pair workflow.")
    p.add_argument("--pre_image", type=Path, required=True, help="Path to one pre-disaster image")
    p.add_argument("--post_image", type=Path, required=True, help="Path to matching post-disaster image")
    p.add_argument("--run_id", type=str, default="sanity1", help="Run id used under --out_root")
    p.add_argument("--out_root", type=Path, default=Path("outputs/driver_runs"), help="Driver output root")
    p.add_argument(
        "--tile_id",
        type=str,
        default="",
        help="Optional tile id override. If omitted, inferred from pre image stem.",
    )

    # Stage-1 args
    p.add_argument(
        "--stage1_script",
        type=Path,
        default=Path("stage1/SAM3_Final_20260226/scripts/run_sam3_building_infer.py"),
        help="Path to Stage-1 run script",
    )
    p.add_argument("--stage1_backend", type=str, default="transformers", choices=["meta", "transformers"])
    p.add_argument("--stage1_device", type=str, default="cuda:0")
    p.add_argument("--stage1_prompt", type=str, default="building")
    p.add_argument("--stage1_min_size", type=int, default=100)
    p.add_argument("--stage1_output_style", type=str, default="notebook", choices=["notebook", "tiled"])
    p.add_argument("--stage1_tile_size", type=int, default=0, help="Optional Stage1 tiling size (0 disables)")
    p.add_argument("--stage1_overlap", type=int, default=64, help="Stage1 tile overlap when tiling is enabled")
    p.add_argument("--stage1_batch_size", type=int, default=1, help="Stage1 inference batch size")

    # Shared artifact args
    p.add_argument("--shared_script", type=Path, default=Path("scripts/generate_shared_instance_subimages.py"))
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--ring_radius_px", type=int, default=48)
    p.add_argument("--shared_num_workers", type=int, default=4)
    p.add_argument("--shared_chunk_size", type=int, default=100)
    p.add_argument("--shared_log_every", type=int, default=50)

    # Stage-2a args
    p.add_argument("--stage2a_build_csv_script", type=Path, default=Path("scripts/build_stage2a_infer_csv.py"))
    p.add_argument("--stage2a_infer_script", type=Path, default=Path("scripts/infer_stage2a.py"))
    p.add_argument(
        "--stage2a_ckpt",
        type=Path,
        default=Path("models/stage2a/stage2a_best_model.pt"),
        help="Stage-2a checkpoint path",
    )
    p.add_argument("--stage2a_batch_size", type=int, default=64)
    p.add_argument("--stage2a_num_workers", type=int, default=4)
    p.add_argument("--stage2a_device", type=str, default="cuda")

    # Stage-2b args
    p.add_argument("--stage2b_infer_script", type=Path, default=Path("scripts/infer_stage2_ensemble.py"))
    p.add_argument(
        "--stage2b_ckpts",
        type=str,
        default=(
            "models/stage2b/inference0.7273.pt,"
            "models/stage2b/inference0.7066_seed9999.pt,"
            "models/stage2b/inference0.7034_seed7777.pt"
        ),
        help="Comma-separated 3 checkpoint paths",
    )
    p.add_argument(
        "--stage2b_configs",
        type=str,
        default=(
            "configs/stage2b/run019_seed2025_train_config.json,"
            "configs/stage2b/seed9999_train_config.json,"
            "configs/stage2b/seed7777_train_config.json"
        ),
        help="Comma-separated 3 train_config paths",
    )
    p.add_argument("--stage2b_weights", type=str, default="4,3,2")
    p.add_argument(
        "--stage2b_calibration_dirs",
        type=str,
        default=(
            "calibration/calibration_run019_r48,"
            "calibration/calibration_seed9999_r48,"
            "calibration/calibration_seed7777_r48"
        ),
    )
    p.add_argument("--stage2b_calibration_method", type=str, default="temperature", choices=["none", "temperature", "vector"])
    p.add_argument("--stage2b_batch_size", type=int, default=64)
    p.add_argument("--stage2b_num_workers", type=int, default=4)
    p.add_argument("--stage2b_device", type=str, default="cuda")
    p.add_argument("--stage2b_print_examples", type=int, default=10)
    p.add_argument("--stage2b_log_every_steps", type=int, default=10)

    # Presenter args
    p.add_argument("--present_script", type=Path, default=Path("scripts/present_instance_results.py"))
    p.add_argument("--top_k_uncertain", type=int, default=30)
    p.add_argument("--print_top_n", type=int, default=15)
    p.add_argument("--visualize_script", type=Path, default=Path("scripts/visualize_stage2_overlays.py"))
    p.add_argument("--visualize_max_outputs", type=int, default=100)
    p.add_argument("--visualize_fill_opacity", type=float, default=0.5)
    p.add_argument("--visualize_gt_labels", type=str, default="", help="Optional gt label filter for visualization")

    # Runtime behavior
    p.add_argument("--python_bin", type=str, default=sys.executable, help="Python interpreter")
    p.add_argument("--overwrite_run_dir", action="store_true", help="Delete existing run directory if present")
    p.add_argument("--dry_run", action="store_true", help="Print commands only, do not execute")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable full stage logs. Default is quiet mode.",
    )
    return p.parse_args()


def infer_tile_id(pre_image: Path, override: str):
    if override:
        return override
    stem = pre_image.stem
    if stem.endswith("_pre_disaster"):
        return stem[: -len("_pre_disaster")]
    return stem


def run_cmd(cmd, dry_run=False, verbose=False):
    if verbose or dry_run:
        print("[cmd]", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    cmd_str = [str(x) for x in cmd]
    if verbose:
        subprocess.run(cmd_str, check=True)
    else:
        subprocess.run(
            cmd_str,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def run_cmd_summary_only(cmd, dry_run=False, verbose=False, prefixes=("[summary]",)):
    if verbose or dry_run:
        print("[cmd]", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    cmd_str = [str(x) for x in cmd]
    if verbose:
        subprocess.run(cmd_str, check=True)
        return
    proc = subprocess.run(
        cmd_str,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        if any(line.startswith(p) for p in prefixes):
            print(line)
    for line in proc.stderr.splitlines():
        if any(line.startswith(p) for p in prefixes):
            print(line)


def main():
    args = parse_args()
    if not args.pre_image.exists():
        raise FileNotFoundError(f"Missing pre image: {args.pre_image}")
    if not args.post_image.exists():
        raise FileNotFoundError(f"Missing post image: {args.post_image}")
    if not args.stage2a_ckpt.exists():
        raise FileNotFoundError(f"Missing Stage-2a checkpoint: {args.stage2a_ckpt}")

    tile_id = infer_tile_id(args.pre_image, args.tile_id)
    run_dir = args.out_root / args.run_id
    if run_dir.exists() and args.overwrite_run_dir:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Prepare a local pair with canonical naming for token-based matching.
    pair_dir = run_dir / "pair_inputs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    pre_link = pair_dir / f"{tile_id}_pre_disaster.png"
    post_link = pair_dir / f"{tile_id}_post_disaster.png"
    for dst in [pre_link, post_link]:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    pre_link.symlink_to(args.pre_image.resolve())
    post_link.symlink_to(args.post_image.resolve())

    stage1_out = run_dir / "stage1"
    shared_out = run_dir / "shared_instances_r48"
    stage2a_input_csv = run_dir / "stage2a_infer_input.csv"
    stage2a_out_csv = run_dir / "stage2a_predictions.csv"
    stage2b_out_jsonl = run_dir / "stage2b_ensemble.jsonl"
    presented_csv = run_dir / "instance_results_presented.csv"
    uncertain_csv = run_dir / "instance_results_top_uncertain.csv"
    vis_dir = run_dir / "vis_instance_level"

    py = args.python_bin

    # 1) Stage-1
    run_cmd(
        [
            py,
            args.stage1_script,
            "--input",
            pair_dir,
            "--output",
            stage1_out,
            "--pattern",
            f"{tile_id}_pre_disaster.png",
            "--max-images",
            "1",
            "--prompt",
            args.stage1_prompt,
            "--min-size",
            str(args.stage1_min_size),
            "--output-style",
            args.stage1_output_style,
            "--batch-size",
            str(args.stage1_batch_size),
            "--device",
            args.stage1_device,
            "--backend",
            args.stage1_backend,
        ]
        + (
            [
                "--tile-size",
                str(args.stage1_tile_size),
                "--overlap",
                str(args.stage1_overlap),
            ]
            if args.stage1_tile_size and args.stage1_tile_size > 0
            else []
        ),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # 2) Shared subimages
    run_cmd(
        [
            py,
            args.shared_script,
            "--stage1_labels_dir",
            stage1_out / "labels",
            "--pre_images_dir",
            pair_dir,
            "--post_images_dir",
            pair_dir,
            "--out_root",
            shared_out,
            "--crop_size",
            str(args.crop_size),
            "--ring_radius_px",
            str(args.ring_radius_px),
            "--strict_images",
            "--num_workers",
            str(args.shared_num_workers),
            "--chunk_size",
            str(args.shared_chunk_size),
            "--log_every",
            str(args.shared_log_every),
        ],
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # 3) Stage-2a csv adapter
    run_cmd(
        [
            py,
            args.stage2a_build_csv_script,
            "--shared_csv",
            shared_out / "shared_instance_samples.csv",
            "--out_csv",
            stage2a_input_csv,
            "--crop_col",
            "pre_crop",
            "--mask_col",
            "mask_M",
        ],
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # 4) Stage-2a inference
    run_cmd(
        [
            py,
            args.stage2a_infer_script,
            "--input_csv",
            stage2a_input_csv,
            "--ckpt",
            args.stage2a_ckpt,
            "--out_csv",
            stage2a_out_csv,
            "--batch_size",
            str(args.stage2a_batch_size),
            "--num_workers",
            str(args.stage2a_num_workers),
            "--device",
            args.stage2a_device,
            "--print_examples",
            "10",
        ],
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    # 5) Stage-2b inference
    cmd_stage2b = [
        py,
        args.stage2b_infer_script,
        "--csv",
        shared_out / "shared_instance_samples.csv",
        "--ckpts",
        args.stage2b_ckpts,
        "--weights",
        args.stage2b_weights,
        "--configs",
        args.stage2b_configs,
        "--calibration_method",
        args.stage2b_calibration_method,
        "--out_jsonl",
        stage2b_out_jsonl,
        "--batch_size",
        str(args.stage2b_batch_size),
        "--num_workers",
        str(args.stage2b_num_workers),
        "--device",
        args.stage2b_device,
        "--print_examples",
        str(args.stage2b_print_examples),
        "--log_every_steps",
        str(args.stage2b_log_every_steps),
    ]
    if args.stage2b_calibration_dirs:
        cmd_stage2b += ["--calibration_dirs", args.stage2b_calibration_dirs]
    run_cmd(cmd_stage2b, dry_run=args.dry_run, verbose=args.verbose)

    # 6) Presentation driver
    run_cmd_summary_only(
        [
            py,
            args.present_script,
            "--shared_csv",
            shared_out / "shared_instance_samples.csv",
            "--stage2a_csv",
            stage2a_out_csv,
            "--stage2b_jsonl",
            stage2b_out_jsonl,
            "--out_csv",
            presented_csv,
            "--out_top_uncertain_csv",
            uncertain_csv,
            "--top_k_uncertain",
            str(args.top_k_uncertain),
            "--print_top_n",
            str(args.print_top_n),
        ],
        dry_run=args.dry_run,
        verbose=args.verbose,
        prefixes=("[summary]",),
    )

    # 7) Instance-level visualization
    cmd_vis = [
        py,
        args.visualize_script,
        "--pred_jsonl",
        stage2b_out_jsonl,
        "--csv",
        shared_out / "shared_instance_samples.csv",
        "--stage2a_csv",
        stage2a_out_csv,
        "--out_dir",
        vis_dir,
        "--max_outputs",
        str(args.visualize_max_outputs),
        "--fill_opacity",
        str(args.visualize_fill_opacity),
    ]
    if args.visualize_gt_labels:
        cmd_vis += ["--gt_labels", args.visualize_gt_labels]
    run_cmd(cmd_vis, dry_run=args.dry_run, verbose=args.verbose)

    print("[done] run_dir:", run_dir)
    print("[done] stage1_out:", stage1_out)
    print("[done] shared_csv:", shared_out / "shared_instance_samples.csv")
    print("[done] stage2a_out:", stage2a_out_csv)
    print("[done] stage2b_out:", stage2b_out_jsonl)
    print("[done] presented_csv:", presented_csv)
    print("[done] top_uncertain_csv:", uncertain_csv)
    print("[done] vis_dir:", vis_dir)


if __name__ == "__main__":
    main()
