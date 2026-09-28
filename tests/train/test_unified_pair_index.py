import json
import runpy
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from sam3.train.data.unified_pair_loader import UnifiedPositivePairFromSQLite

_builder = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/build_steersam_pairs.py"))
PairBuilder = _builder["PairBuilder"]
finalize_index = _builder["finalize_index"]


def create_schema(conn):
    _builder["create_schema"](conn)
    # These unit fixtures intentionally contain only a few known records.
    finalize_index(conn)
phrasecut_candidates = _builder["phrasecut_candidates"]
remove_cross_split_duplicates = _builder["remove_cross_split_duplicates"]
validate_box = _builder["validate_box"]


class _Tokenizer:
    def encode(self, text):
        return text.split()


def _candidate(source, record_id, targets, split="train", text="cat"):
    return {
        "source": source,
        "split": split,
        "source_record_id": record_id,
        "image_key": "coco:1",
        "image_path": "image.jpg",
        "original_image_id": 1,
        "original_category_id": 1,
        "text": text,
        "width": 20,
        "height": 20,
        "is_exhaustive": True,
        "is_pixel_exhaustive": True,
        "raw_targets": targets,
        "target_format": "coco",
    }


def test_pair_conflicts_filters_and_loader(tmp_path):
    Image.new("RGB", (20, 20)).save(tmp_path / "image.jpg")
    index = tmp_path / "pairs.sqlite"
    polygon1 = [[1, 1, 8, 1, 8, 8, 1, 8]]
    polygon2 = [[10, 10, 18, 10, 18, 18, 10, 18]]
    one = [(1, [1, 1, 7, 7], polygon1)]
    two = one + [(2, [10, 10, 8, 8], polygon2)]
    with sqlite3.connect(index) as conn:
        create_schema(conn)
        builder = PairBuilder(tmp_path, conn, _Tokenizer())
        builder.add(_candidate("coco", "one", one))
        builder.add(_candidate("lvis", "two", two))  # more targets wins
        builder.add(_candidate("coco", "exact", two))  # exact duplicate
        builder.add(_candidate("phrasecut", "conflict", [
            (3, [2, 2, 6, 6], [[2, 2, 8, 2, 8, 8, 2, 8]]), two[1]
        ]))  # equal count, different masks: retain both
        builder.add(_candidate("refcoco", "bad_mask", [(4, [1, 1, 7, 7], [])], text="dog"))
        builder.add(_candidate("refcoco", "long", one, text=" ".join(["word"] * 31)))
        rows = conn.execute("SELECT source, target_count FROM pairs ORDER BY source").fetchall()
        assert rows == [("lvis", 2), ("phrasecut", 2)]
        payload = json.loads(conn.execute("SELECT payload FROM pairs WHERE source='lvis'").fetchone()[0])
        assert payload["source_dataset"] == "lvis"
        assert payload["original_split"] == "train"
        assert builder.report[("coco", "train")]["reasons"]["replaced_by_more_targets"] == 1
        assert builder.report[("coco", "train")]["reasons"]["exact_duplicate"] == 1
        assert builder.retained_tied_conflicts == 1
        assert builder.report[("refcoco", "train")]["reasons"]["text_over_32_bpe"] == 1

    loader = UnifiedPositivePairFromSQLite(str(index), max_train_pairs={"lvis": 1, "phrasecut": 0})
    loader_again = UnifiedPositivePairFromSQLite(str(index), max_train_pairs={"lvis": 1, "phrasecut": 0})
    assert loader.getDatapointIds() == loader_again.getDatapointIds()
    assert loader.source_counts == {"lvis": 1}
    assert len(loader.getDatapointIds()) == 1
    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(loader.getDatapointIds()[0])
    assert queries[0]["query_text"] == "cat"
    assert queries[0]["object_ids_output"] == [0, 1]
    assert len(annotations) == 2
    assert len(annotations[0]["segmentation"]["counts"]) > 0


def test_phrasecut_annotation_ids_are_not_region_count(tmp_path):
    task = {
        "task_id": "7__2", "image_id": 7, "ann_ids": [100], "phrase": "two cats",
        "instance_boxes": [[1, 1, 7, 7], [10, 10, 8, 8]],
        "Polygons": [
            [[[1, 1], [8, 1], [8, 8], [1, 8]]],
            [[[10, 10], [18, 10], [18, 18], [10, 18]]],
        ],
    }
    refer = tmp_path / "refer_train.json"
    refer.write_text(json.dumps([task]))
    metadata = {7: {"split": "train", "width": 20, "height": 20}}
    candidate = next(phrasecut_candidates(refer, metadata))
    assert candidate["source_annotation_ids"] == [100]
    assert len(candidate["raw_targets"]) == 2
    assert candidate["raw_targets"][0][0] == "7__2:0"


def test_held_out_exact_pair_removes_only_training_copy(tmp_path):
    Image.new("RGB", (20, 20)).save(tmp_path / "image.jpg")
    polygon = [[1, 1, 8, 1, 8, 8, 1, 8]]
    target = [(1, [1, 1, 7, 7], polygon)]
    with sqlite3.connect(tmp_path / "pairs.sqlite") as conn:
        create_schema(conn)
        builder = PairBuilder(tmp_path, conn, _Tokenizer())
        builder.add(_candidate("coco", "train", target))
        builder.add(_candidate("refcoco", "val", target, split="val"))
        builder.add(_candidate("refcoco", "different", target, text="the cat"))
        assert remove_cross_split_duplicates(conn) == 1
        rows = conn.execute("SELECT split, text FROM (SELECT split, json_extract(payload, '$.text') text FROM pairs)").fetchall()
        assert sorted(rows) == [("train", "the cat"), ("val", "cat")]


def test_small_phrasecut_box_overhang_is_clipped_but_large_error_is_rejected():
    assert validate_box([0, 129.17, 500.17, 170.17], 500, 300) == pytest.approx(
        [0.0, 129.17, 500.0, 170.17]
    )
    with pytest.raises(ValueError, match="invalid_box"):
        validate_box([0, 0, 120, 10], 100, 100)


def test_per_source_cap_is_stable_after_build(tmp_path):
    Image.new("RGB", (20, 20)).save(tmp_path / "image.jpg")
    target = [(1, [1, 1, 7, 7], [[1, 1, 8, 1, 8, 8, 1, 8]])]
    index = tmp_path / "pairs.sqlite"
    with sqlite3.connect(index) as conn:
        create_schema(conn)
        builder = PairBuilder(tmp_path, conn, _Tokenizer())
        for i in range(8):
            builder.add(_candidate("coco", str(i), target, text=f"category {i}"))
    first = UnifiedPositivePairFromSQLite(str(index), max_train_pairs={"coco": 3}, selection_seed=7)
    repeat = UnifiedPositivePairFromSQLite(str(index), max_train_pairs={"coco": 3}, selection_seed=7)
    assert len(first.getDatapointIds()) == 3
    assert first.getDatapointIds() == repeat.getDatapointIds()
    assert first.available_source_counts == {"coco": 8}


def test_partial_and_unfinished_indexes_require_safe_handling(tmp_path):
    Image.new("RGB", (20, 20)).save(tmp_path / "image.jpg")
    path = tmp_path / "pairs.sqlite"
    with sqlite3.connect(path) as conn:
        create_schema(conn)
        PairBuilder(tmp_path, conn, _Tokenizer()).add(_candidate(
            "coco", "one", [(1, [1, 1, 7, 7], [[1, 1, 8, 1, 8, 8, 1, 8]])]
        ))
        finalize_index(conn, max_source_pairs=10)
    with pytest.raises(ValueError, match="Partial pair index"):
        UnifiedPositivePairFromSQLite(str(path))
    assert UnifiedPositivePairFromSQLite(str(path), allow_partial_index=True).build_state == "partial"
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE metadata SET value='building' WHERE key='build_state'")
    with pytest.raises(ValueError, match="Unfinished"):
        UnifiedPositivePairFromSQLite(str(path), allow_partial_index=True)


def test_legacy_index_audit_compatibility(tmp_path):
    path = tmp_path / "pairs.sqlite"
    with sqlite3.connect(path) as conn:
        create_schema(conn)
        conn.execute("DELETE FROM metadata WHERE key='build_state'")
    with pytest.raises(ValueError, match="adjacent"):
        UnifiedPositivePairFromSQLite(str(path))
    audit = path.with_suffix('.audit.json')
    audit.write_text(json.dumps({"schema_version": 1, "partial_build_max_source_pairs": 10}))
    with pytest.raises(ValueError, match="Partial"):
        UnifiedPositivePairFromSQLite(str(path))
    audit.write_text(json.dumps({"schema_version": 1, "partial_build_max_source_pairs": None}))
    # Completeness is accepted, then the empty fixture fails selection.
    with pytest.raises(ValueError, match="No pairs selected"):
        UnifiedPositivePairFromSQLite(str(path))
