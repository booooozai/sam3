# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Read prevalidated SteerSAM image/query pairs without loading source JSONs.

The builder writes one SQLite index. Each DataLoader worker opens its own
read-only connection after fork/spawn; no connection is pickled or shared.
"""

import hashlib
import heapq
import json
import os
import sqlite3
from pathlib import Path
import torch

from sam3.train.data.pair_contract import SOURCES


class UnifiedPositivePairFromSQLite:
    def __init__(
        self,
        annotation_file: str,
        split: str = "train",
        source: str | None = None,
        max_train_pairs: dict | None = None,
        selection_seed: int = 123,
        allow_partial_index: bool = False,
    ):
        if source is not None and source not in SOURCES:
            raise ValueError(f"Unknown source {source!r}")
        if split not in ("train", "val", "test", "testA", "testB"):
            raise ValueError(f"Unknown split {split!r}")
        if max_train_pairs and split != "train":
            raise ValueError("max_train_pairs only applies to the train split")
        self.annotation_file = os.path.abspath(annotation_file)
        self.split = split
        self.source = source
        self.selection_seed = int(selection_seed)
        self.max_train_pairs = dict(max_train_pairs or {})
        unknown = set(self.max_train_pairs) - set(SOURCES)
        if unknown:
            raise ValueError(f"Unknown source caps: {sorted(unknown)}")
        for name, cap in self.max_train_pairs.items():
            if cap is not None and (not isinstance(cap, int) or cap < 0):
                raise ValueError(f"{name} cap must be null or a nonnegative integer")

        self._conn = None
        self._pid = None
        self._last_record = None
        with sqlite3.connect(f"file:{self.annotation_file}?mode=ro", uri=True) as conn:
            schema = conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if schema is None or schema[0] != "1":
                raise ValueError("Unsupported or incomplete SteerSAM pair index")
            state = conn.execute("SELECT value FROM metadata WHERE key='build_state'").fetchone()
            if state is None:
                # Existing validated indexes predate the embedded build marker.
                # Read their adjacent audit without modifying the index itself.
                audit_path = Path(self.annotation_file).with_suffix('.audit.json')
                if not audit_path.is_file():
                    raise ValueError("Legacy index requires its adjacent .audit.json to verify completeness")
                with audit_path.open() as handle:
                    audit = json.load(handle)
                if audit.get('schema_version') != 1 or 'partial_build_max_source_pairs' not in audit:
                    raise ValueError("Legacy index audit is missing build-completeness metadata")
                build_state = 'complete' if audit['partial_build_max_source_pairs'] is None else 'partial'
            else:
                build_state = state[0]
            if build_state not in ('complete', 'partial'):
                raise ValueError(f"Unfinished or unsupported pair index: {build_state}")
            if build_state == 'partial' and not allow_partial_index:
                raise ValueError("Partial pair index is for smoke tests only; set allow_partial_index=True explicitly")
            self.build_state = build_state
            sql = "SELECT id, pair_id, source FROM pairs WHERE split=?"
            params = [split]
            if source is not None:
                sql += " AND source=?"
                params.append(source)
            rows = conn.execute(sql, params).fetchall()

        by_source = {name: [] for name in SOURCES}
        for row_id, pair_id, row_source in rows:
            by_source[row_source].append((row_id, pair_id))
        selected = []
        self.available_source_counts = {}
        self.source_counts = {}
        for name in SOURCES:
            candidates = by_source[name]
            if candidates:
                self.available_source_counts[name] = len(candidates)
            cap = self.max_train_pairs.get(name) if split == "train" else None
            if cap == 0:
                candidates = []
            elif cap is not None and cap < len(candidates):
                candidates = heapq.nsmallest(
                    cap,
                    candidates,
                    key=lambda item: hashlib.blake2b(
                        f"{self.selection_seed}:{item[1]}".encode(), digest_size=16
                    ).digest(),
                )
            selected.extend(row_id for row_id, _ in candidates)
            if candidates:
                self.source_counts[name] = len(candidates)
        self.ids = sorted(selected)
        if not self.ids:
            raise ValueError(f"No pairs selected from {self.annotation_file} ({split})")
        print(
            f"SteerSAM {split} pair counts: selected={self.source_counts}, "
            f"eligible={self.available_source_counts}, "
            f"caps={self.max_train_pairs if split == 'train' else 'not applied'}"
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_conn"] = None
        state["_pid"] = None
        state["_last_record"] = None
        return state

    def _connection(self):
        pid = os.getpid()
        if self._conn is None or self._pid != pid:
            self._conn = sqlite3.connect(
                f"file:{self.annotation_file}?mode=ro", uri=True, check_same_thread=False
            )
            self._pid = pid
            self._last_record = None
        return self._conn

    def _record(self, row_id):
        if self._last_record is not None and self._last_record[0] == row_id:
            return self._last_record[1]
        row = self._connection().execute(
            "SELECT payload FROM pairs WHERE id=?", (int(row_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"Pair row {row_id} not found")
        record = json.loads(row[0])
        self._last_record = (row_id, record)
        return record

    def getDatapointIds(self):
        return self.ids

    def loadImagesFromDatapoint(self, idx):
        r = self._record(idx)
        return [{
            "id": 0,
            "file_name": r["image_path"],
            "original_img_id": r["original_image_id"],
            "coco_img_id": int(idx),
        }]

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        r = self._record(idx)
        width, height = r["width"], r["height"]
        annotations = []
        for i, target in enumerate(r["targets"]):
            x, y, w, h = target["bbox"]
            segmentation = target["segmentation"]
            annotations.append({
                "id": i,
                "object_id": i,
                "image_id": 0,
                "bbox": torch.tensor([x / width, y / height, w / width, h / height], dtype=torch.float32),
                "area": float(target["area"]),
                "segmentation": segmentation,
                "is_crowd": False,
                "source": r["source"],
            })
        query = {
            "id": 0,
            "original_cat_id": r.get("original_category_id", -1),
            "object_ids_output": list(range(len(annotations))),
            "query_text": r["text"],
            "query_processing_order": 0,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": bool(r["is_exhaustive"]),
            "is_pixel_exhaustive": bool(r["is_pixel_exhaustive"]),
        }
        return [query], annotations
