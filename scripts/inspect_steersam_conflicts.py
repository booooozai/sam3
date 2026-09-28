"""Export equal-target-count, different-annotation pair groups for audit."""

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/steersam_pairs/pairs.sqlite")
    parser.add_argument("--output", default="data/steersam_pairs/pairs.conflicts.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    group_count = 0
    pair_count = 0
    with sqlite3.connect(f"file:{Path(args.index).resolve()}?mode=ro", uri=True) as conn:
        groups = conn.execute("""
            SELECT split,image_key,normalized_text,target_count
            FROM pairs
            GROUP BY split,image_key,normalized_text,target_count
            HAVING COUNT(DISTINCT signature)>1
        """)
        with output.open("w") as handle:
            for split, image_key, normalized_text, target_count in groups:
                # A second cursor is needed while the group cursor is active.
                rows = conn.execute("""
                    SELECT pair_id,source,signature,payload FROM pairs
                    WHERE split=? AND image_key=? AND normalized_text=? AND target_count=?
                    ORDER BY pair_id
                """, (split, image_key, normalized_text, target_count)).fetchall()
                entries = []
                for pair_id, source, signature, payload in rows:
                    record = json.loads(payload)
                    entries.append({
                        "pair_id": pair_id,
                        "source_dataset": source,
                        "signature": signature,
                        "image_path": record["image_path"],
                        "target_annotation_ids": [t["original_annotation_id"] for t in record["targets"]],
                    })
                handle.write(json.dumps({
                    "split": split, "image_key": image_key,
                    "normalized_text": normalized_text, "target_count": target_count,
                    "pairs": entries,
                }, ensure_ascii=False) + "\n")
                group_count += 1
                pair_count += len(entries)
    print(f"Wrote {group_count} conflict groups ({pair_count} pairs) to {output}")


if __name__ == "__main__":
    main()
