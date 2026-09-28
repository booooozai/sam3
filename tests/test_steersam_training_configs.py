"""CPU checks for the six-source training and internal-validation configs."""

import inspect
from pathlib import Path

import pytest
from hydra.utils import get_object
from omegaconf import DictConfig, ListConfig

from sam3.train.data.pair_contract import SOURCES
from scripts.report_steersam_selection import load_training_config


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "sam3/train/configs/steersam"
CONFIG_NAMES = (
    "steersam_six_dataset",
    "steersam_six_dataset_mini",
    *(f"steersam_internal_val_{source}" for source in SOURCES),
)


def _check_targets(node):
    """Resolve configured callables without constructing a model or trainer."""
    if isinstance(node, DictConfig):
        if "_target_" in node:
            assert callable(get_object(node._target_))
        for value in node.values():
            _check_targets(value)
    elif isinstance(node, ListConfig):
        for value in node:
            _check_targets(value)


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_training_configs_resolve_and_keep_strategy_two(name):
    cfg = load_training_config(CONFIG_ROOT / f"{name}.yaml")
    _check_targets(cfg)
    model = cfg.trainer.model
    inspect.signature(get_object(model._target_)).bind(
        **{key: value for key, value in model.items() if not key.startswith("_")}
    )
    assert model.enable_steering and model.enable_patch_supervision
    assert model.freeze_vision_backbone and model.freeze_language_backbone
    assert model.freeze_sam3_task_modules and not model.freeze_vision_fpn
    assert model.resolution == cfg.scratch.resolution
    assert cfg.scratch.resolution % 14 == 0
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert cfg.trainer.meters is None
    assert cfg.trainer.loss.default.task_loss is None
    assert cfg.trainer.loss.all.task_loss is not None
    for split in ("train", "val"):
        loader = cfg.trainer.data[split].dataset.coco_json_loader
        assert loader._target_.endswith("UnifiedPositivePairFromSQLite")
        assert loader.split == split
        assert cfg.trainer.data[split].dataset.load_segmentation
        kwargs = {key: value for key, value in loader.items() if not key.startswith("_")}
        inspect.signature(get_object(loader._target_)).bind("index.sqlite", **kwargs)
    assert "max_train_pairs" not in cfg.trainer.data.val.dataset.coco_json_loader


def test_selection_report_resolves_inherited_mini():
    cfg = load_training_config(CONFIG_ROOT / "steersam_six_dataset_mini.yaml")
    assert cfg.scratch.pair_index.endswith("pairs.sqlite")
    assert cfg.scratch.pair_selection_seed == 123
    assert set(cfg.scratch.max_train_pairs) == set(SOURCES)
    assert all(cap == 200 for cap in cfg.scratch.max_train_pairs.values())
    assert cfg.trainer.data.train.dataset.coco_json_loader.max_train_pairs == cfg.scratch.max_train_pairs


@pytest.mark.parametrize("source", SOURCES)
def test_internal_validation_configs_still_resolve(source):
    cfg = load_training_config(CONFIG_ROOT / f"steersam_internal_val_{source}.yaml")
    assert cfg.trainer.mode == "val"
    assert cfg.trainer.data.val.dataset.coco_json_loader.source == source
    assert cfg.trainer.checkpoint.resume_from == (
        f"{cfg.launcher.experiment_log_dir}/checkpoints/checkpoint.pt"
    )
    assert cfg.trainer.meters is None
