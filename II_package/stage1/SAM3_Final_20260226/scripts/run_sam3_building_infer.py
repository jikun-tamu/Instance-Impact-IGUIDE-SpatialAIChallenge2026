# Sample command: python run_sam3_building_infer.py --input /media/data/building_instance_tamu/test/images --output /media/data/building_instance_tamu/sam3/test_260225 --prompt "building" --min-size 100 --tile-size 512 --overlap 64 --regularize geoai --epsilon 2.0 --metadata /path/to/metadata.csv --backend meta --device cuda:0 --checkpoint /path/to/checkpoint.pth

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sam3_final.pipeline import PipelineConfig, run_pipeline
from sam3_final.utils import get_env_var


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAM3 pre-disaster building footprint extraction")
    p.add_argument("--input", required=True, help="Input image or directory")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--prompt", default="building", help="SAM3 text prompt")
    p.add_argument("--min-size", type=int, default=100, help="Minimum mask size")
    p.add_argument("--tile-size", type=int, default=None, help="Tile size in pixels")
    p.add_argument("--overlap", type=int, default=0, help="Tile overlap in pixels")
    p.add_argument(
        "--regularize",
        default="none",
        choices=["none", "simplify", "min_rot_rect", "geoai"],
        help="Regularization method",
    )
    p.add_argument("--epsilon", type=float, default=2.0, help="Regularization epsilon")
    p.add_argument("--metadata", default=None, help="Optional metadata table (csv/json) for georef")
    p.add_argument("--no-annotations", action="store_true", help="Disable all annotation PNG output")
    p.add_argument("--tile-annotations", action="store_true", help="Save per-tile annotation PNGs")
    p.add_argument("--full-annotation", action="store_true", help="Save full-size annotation PNG")
    p.add_argument("--output-style", default="notebook", choices=["notebook", "tiled"], help="Output style")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    p.add_argument("--max-images", type=int, default=None, help="Max number of images to process")
    p.add_argument("--pattern", default="*_pre_disaster.png", help="Filename glob pattern")
    p.add_argument("--no-masks", action="store_true", help="Disable mask output (still needed for run)")
    p.add_argument("--no-scores", action="store_true", help="Disable score TIFF output")
    p.add_argument("--no-polygons", action="store_true", help="Skip polygonize/regularize/export")
    p.add_argument("--clear-cache-every", type=int, default=0, help="Clear CUDA cache every N tiles (0=disabled)")
    p.add_argument("--backend", default="meta", help="SAM3 backend (meta|transformers)")
    p.add_argument("--device", default=None, help="Device override (e.g., cuda:0)")
    p.add_argument("--checkpoint", default=None, help="Local checkpoint path")
    p.add_argument("--no-hf", action="store_true", help="Do not load weights from Hugging Face")
    p.add_argument("--exts", default="png,jpg,jpeg,tif,tiff", help="Comma-separated extensions")
    return p


def main() -> None:
    args = build_parser().parse_args()

    hf_token = get_env_var("HF_TOKEN")

    cfg = PipelineConfig(
        input_path=args.input,
        output_dir=args.output,
        prompt=args.prompt,
        min_size=args.min_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        regularize_method=("geoai" if args.regularize == "geoai" else args.regularize),
        epsilon=args.epsilon,
        use_geoai=(args.regularize == "geoai"),
        metadata_path=args.metadata,
        save_masks=not args.no_masks,
        save_scores=not args.no_scores,
        save_annotations=not args.no_annotations,
        tile_annotations=args.tile_annotations,
        full_annotation=(args.full_annotation and not args.no_annotations),
        output_style=args.output_style,
        batch_size=args.batch_size,
        max_images=args.max_images,
        pattern=args.pattern,
        run_polygons=not args.no_polygons,
        clear_cache_every=args.clear_cache_every,
        sam3_backend=args.backend,
        sam3_device=args.device,
        sam3_checkpoint=args.checkpoint,
        sam3_load_from_hf=not args.no_hf,
        hf_token=hf_token,
        exts=tuple([e.strip() for e in args.exts.split(",") if e.strip()]),
    )

    summary = run_pipeline(cfg)
    out_path = Path(args.output) / "run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
