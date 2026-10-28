import copy
import csv
import json

import pytest

from research.fedgcdr_akf.data import prepare


@pytest.fixture
def toy_data(tmp_path):
    ratings = tmp_path / "ratings"
    ratings.mkdir()
    target = []
    for user in range(8):
        for offset, base in enumerate((10, 20, 30, 40, 50)):
            target.append((f"i{user + offset}", f"u{user}", 5, base + user))
    target.append(("i0", "u0", 5, 999))  # Earliest duplicate wins.
    for name in ("Target", "SourceA", "SourceB"):
        rows = target if name == "Target" else [
            (f"s{user}", f"u{user}", 4, 10 + user) for user in range(7)
        ] + [("future_only", "u0", 5, 1000), ("low_rating", "u1", 2, 12)]
        with (ratings / f"{name}.csv").open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
    cfg = {
        "ratings_dir": str(ratings), "target": "Target", "sources": ["SourceA", "SourceB"],
        "max_rows_per_domain": 0, "max_users": 0, "min_train_interactions": 2,
        "rating_threshold": 4., "train_fraction": .59, "validation_fraction": .2, "seed": 3,
    }
    output = tmp_path / "prepared"
    summary = prepare(cfg, output)
    return cfg, output, summary


def test_temporal_split_source_cutoff_and_training_only_vocabulary(toy_data):
    _, output, summary = toy_data
    data = json.loads((output / "prepared.json").read_text())
    target = data["domains"]["Target"]
    assert summary["users"] == 8
    assert summary["source_overlap_users"] == 7
    assert summary["input_manifest"][0]["duplicate_positive_rows"] == 1
    assert all(row["hash_scope"] == "full_file" for row in summary["input_manifest"])
    for domain in summary["domains"].values():
        assert domain["max_train_timestamp"] <= data["train_cutoff"]
    for source in ("SourceA", "SourceB"):
        assert "future_only" not in data["domains"][source]["items"]
        assert "low_rating" not in data["domains"][source]["items"]
        assert not data["domains"][source]["train"][data["user_ids"].index("u7")]
    for train, validation, test in zip(target["train"], target["validation"], target["test"]):
        assert len(train) == 3
        assert set(train).isdisjoint(validation)
        assert set(train).isdisjoint(test)
        assert set(validation).isdisjoint(test)
    assert summary["domains"]["Target"]["test"]["dropped_unseen_items"] > 0


def test_preparation_is_reproducible_and_does_not_overwrite(toy_data, tmp_path):
    cfg, output, _ = toy_data
    second = tmp_path / "again"
    prepare(cfg, second)
    assert (output / "prepared.json").read_bytes() == (second / "prepared.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(cfg, output)


def test_domain_path_traversal_rejected(toy_data, tmp_path):
    cfg, _, _ = toy_data
    bad = copy.deepcopy(cfg)
    bad["sources"] = ["../SourceA"]
    with pytest.raises(ValueError, match="basenames"):
        prepare(bad, tmp_path / "bad")
