"""Report the post-filter, post-dedup train mixture selected by a YAML config."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from sam3.train.utils.train_utils import register_omegaconf_resolvers

from sam3.train.data.unified_pair_loader import UnifiedPositivePairFromSQLite


def load_training_config(config_path):
    """Resolve Hydra defaults using the same config root as train.py."""
    config_path = Path(config_path).resolve()
    config_root = Path(__file__).resolve().parents[1] / "sam3" / "train"
    config_name = config_path.relative_to(config_root).as_posix()
    if not OmegaConf.has_resolver("times"):
        register_omegaconf_resolvers()
    with initialize_config_dir(config_dir=str(config_root), version_base="1.2"):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="sam3/train/configs/steersam/steersam_six_dataset.yaml")
    parser.add_argument("--index", default=None, help="Override scratch.pair_index (e.g. a CPU smoke index)")
    parser.add_argument("--output", default="data/steersam_pairs/pairs.selection.json")
    args = parser.parse_args()
    cfg = load_training_config(args.config)
    index_path = Path(args.index or cfg.scratch.pair_index).resolve()
    caps = OmegaConf.to_container(cfg.scratch.max_train_pairs, resolve=True)
    seed = int(cfg.scratch.pair_selection_seed)
    selection = UnifiedPositivePairFromSQLite(
        str(index_path), split="train", max_train_pairs=caps, selection_seed=seed,
        allow_partial_index=bool(
            cfg.trainer.data.train.dataset.coco_json_loader.get("allow_partial_index", False)
        ),
    )
    report = {
        "index": str(index_path),
        "config": str(Path(args.config).resolve()),
        "selection_seed": seed,
        "max_train_pairs": caps,
        "eligible_train_pairs": selection.available_source_counts,
        "selected_train_pairs": selection.source_counts,
        "total_selected_train_pairs": len(selection.getDatapointIds()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
