from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .hf_builders import build_and_write_medmcqa, build_and_write_pubmedqa
from .mimic_builder import build_and_write_mimic_esi
from .scale_builders import build_and_write_scale

ROOT = Path(__file__).resolve().parents[1]


def main(default_dataset_id: Optional[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Build a versioned MedJev dataset and write Parquet plus a manifest.")
    parser.add_argument("--dataset", choices=["medmcqa", "pubmedqa", "news2", "gcs", "curb65", "mimic_esi"], default=default_dataset_id)
    parser.add_argument("--registry", type=Path, help="Registry YAML (defaults to data/registry/<dataset>.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, help="Override the default train count for synthetic scales or MIMIC ESI")
    parser.add_argument("--input", type=Path, help="Local MIMIC triage CSV; it is read in place and is never copied")
    parser.add_argument("--output-dir", type=Path, help="Override the ignored processed output directory")
    args = parser.parse_args()
    dataset_id = args.dataset or default_dataset_id
    if not dataset_id:
        parser.error("Pass --dataset or use the source-specific build.py")
    if args.registry:
        registry_path = args.registry.expanduser().resolve()
    else:
        registry_path = ROOT / "data" / "registry" / f"{dataset_id}.yaml"
    if not registry_path.is_file():
        parser.error(f"Registry not found: {registry_path}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else None
    if dataset_id == "medmcqa":
        manifest = build_and_write_medmcqa(registry_path, seed=args.seed, output_dir=output_dir)
    elif dataset_id == "pubmedqa":
        manifest = build_and_write_pubmedqa(registry_path, seed=args.seed, output_dir=output_dir)
    elif dataset_id in {"news2", "gcs", "curb65"}:
        manifest = build_and_write_scale(dataset_id, registry_path, seed=args.seed, count=args.count, output_dir=output_dir)
    else:
        manifest = build_and_write_mimic_esi(registry_path, input_path=args.input, seed=args.seed, count=args.count or 200, output_dir=output_dir)
    print(f"Built {dataset_id}: {manifest['split_counts']} (seed={manifest['seed']})")


if __name__ == "__main__":
    main()
