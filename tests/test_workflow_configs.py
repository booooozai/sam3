"""CPU coverage of entry-point contracts that model unit tests cannot catch."""

import inspect
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module
from hydra.utils import get_class
from omegaconf import OmegaConf

from scripts.eval.steersam import run_official_coco_o as coco_o
from scripts.eval.steersam import run_official_saco_gold as gold
from sam3.train.data.coco_json_loaders import COCOAllCategoryPairsFromJSON
from sam3.train.utils.train_utils import register_omegaconf_resolvers

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    if not OmegaConf.has_resolver("times"):
        register_omegaconf_resolvers()
    with initialize_config_module(config_module="sam3.train", version_base="1.2"):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    return cfg


@pytest.mark.parametrize("name", [
    "configs/coco/coco2017_steersam_mask_eval",
    "configs/steersam/eval_official_sam3_coco_672",
    "configs/steersam/eval_official_sam3_coco_1008",
    "configs/steersam/eval_official_sam3_lvis_672",
    "configs/steersam/eval_official_sam3_lvis_1008",
])
def test_formal_eval_loader_constructor_and_meters(name):
    cfg = config(name)
    loader = cfg.trainer.data.val.dataset.coco_json_loader
    kwargs = {k: v for k, v in loader.items() if not k.startswith("_")}
    inspect.signature(get_class(loader._target_)).bind("annotations.json", **kwargs)
    for meters in cfg.trainer.meters.val.values():
        assert set(meters) == {"segmentation", "detection"}


def test_all_category_pair_image_cap(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({
        "images": [{"id": i, "width": 8, "height": 8, "file_name": f"{i}.jpg"} for i in (1, 2)],
        "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
        "annotations": [],
    }))
    loader = COCOAllCategoryPairsFromJSON(str(path), max_images=1)
    assert len(loader.getDatapointIds()) == 2
    assert {loader.loadImagesFromDatapoint(i)[0]["original_img_id"] for i in loader.getDatapointIds()} == {1}


def test_coco_o_aggregation_uses_last_jsonl_record(tmp_path):
    for domain in coco_o.DOMAINS:
        path = tmp_path / domain / "logs/val_stats.json"
        path.parent.mkdir(parents=True)
        key = f"Meters_train/val_{domain}/detection/coco_eval_bbox_AP"
        path.write_text(json.dumps({key: 0.1}) + "\n" + json.dumps({key: 0.2}) + "\n")
        assert coco_o.read_last_stats(path)[key] == 0.2
    assert coco_o.aggregate(tmp_path) == 0


def test_coco_o_missing_metrics_fail(tmp_path):
    path = tmp_path / "cartoon/logs/val_stats.json"
    path.parent.mkdir(parents=True)
    path.write_text('{}\n')
    assert coco_o.aggregate(tmp_path) == 1


def test_coco_o_dry_run_never_launches_or_reads_results(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run", "--output-root", str(tmp_path)])
    monkeypatch.setattr(coco_o.subprocess, "run", lambda *a, **k: pytest.fail("unexpected launch"))
    monkeypatch.setattr(coco_o, "aggregate", lambda *a: pytest.fail("unexpected result read"))
    coco_o.main()


def test_coco_o_custom_output_is_forwarded(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sys, "argv", ["runner", "--output-root", str(tmp_path)])
    monkeypatch.setattr(coco_o.subprocess, "run", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(coco_o, "aggregate", lambda p: 0)
    with pytest.raises(SystemExit) as result:
        coco_o.main()
    assert result.value.code == 0
    assert len(calls) == 6
    assert all(c["env"]["COCO_O_OUTPUT_ROOT"] == str(tmp_path) for c in calls)
    monkeypatch.setenv("COCO_O_OUTPUT_ROOT", str(tmp_path))
    for domain in coco_o.DOMAINS:
        cfg = config(f"configs/coco_o_evals/sam3_coco_o_{domain}")
        assert Path(cfg.paths.experiment_log_dir) == tmp_path / domain


def gold_gt_files(root):
    for stem in ("metaclip", "sa1b", "crowded", "fg_food", "fg_sports_equipment", "attributes", "wiki_common"):
        for annotator in ("a", "b", "c"):
            (root / f"gold_{stem}_merged_{annotator}_release_test.json").write_text('{}')


def test_gold_checks_all_annotators_and_forwards_paths(monkeypatch, tmp_path):
    gold_gt_files(tmp_path)
    calls = []
    out = tmp_path / "outputs"
    monkeypatch.setattr(sys, "argv", ["runner", "--gt-folder", str(tmp_path), "--pred-folder", str(out)])
    monkeypatch.setattr(gold.subprocess, "run", lambda *a, **k: calls.append(k))
    gold.main()
    assert len(calls) == 8
    assert calls[0]["env"]["SACO_GOLD_GT_ROOT"] == str(tmp_path)
    assert calls[0]["env"]["SACO_GOLD_OUTPUT_ROOT"] == str(out)
    monkeypatch.setenv("SACO_GOLD_GT_ROOT", str(tmp_path))
    monkeypatch.setenv("SACO_GOLD_OUTPUT_ROOT", str(out))
    cfg = config("configs/gold_image_evals/sam3_gold_image_attributes")
    assert Path(cfg.paths.coco_gt).parent == tmp_path
    assert Path(cfg.paths.experiment_log_dir) == out / "gold_attributes"
    (tmp_path / "gold_attributes_merged_b_release_test.json").unlink()
    with pytest.raises(FileNotFoundError, match="merged_b"):
        gold.main()


@pytest.mark.parametrize("runner", [gold, coco_o])
def test_async_cluster_mode_is_rejected(monkeypatch, runner):
    monkeypatch.setattr(sys, "argv", ["runner", "--use-cluster", "1"])
    with pytest.raises(SystemExit) as result:
        runner.main()
    assert result.value.code == 2
