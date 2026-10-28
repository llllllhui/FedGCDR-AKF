import json

import torch

from research.fedgcdr_akf.run import run
from test_data import toy_data


def test_end_to_end_selects_validation_and_tests_once_per_mode(toy_data, tmp_path):
    data_cfg, prepared, _ = toy_data
    config = {"data": data_cfg, "training": {
        "seed": 4, "device": "cpu", "threads": 1, "embedding_dim": 4,
        "pretrain_rounds": 1, "fusion_rounds": 2, "local_epochs": 1,
        "client_batch_size": 4, "max_history": 3, "max_positives": 3,
        "negatives_per_positive": 1, "learning_rate": .01, "fixed_gamma": .5,
        "fixed_gamma_candidates": [.25, .5, 1.],
        "modes": ["fixed", "global", "user", "item"],
    }}
    output = tmp_path / "run"
    summary = run(config, prepared / "prepared.json", output)
    logs = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    for mode in config["training"]["modes"]:
        validation = [row for row in logs if row.get("mode") == mode and row.get("split") == "validation"]
        tests = [row for row in logs if row.get("mode") == mode and row.get("split") == "test"]
        assert len(validation) == (6 if mode == "fixed" else 2) and len(tests) == 1
        best = max(validation, key=lambda row: (row["metrics"]["ndcg@10"], row["metrics"]["recall@10"]))
        assert tests[0]["selected_round"] == best["round"]
        assert tests[0]["selected_fixed_gamma"] == best["fixed_gamma"]
        assert logs.index(tests[0]) > max(logs.index(row) for row in validation)
        assert (output / f"{mode}.pt").is_file()
        saved = torch.load(output / f"{mode}.pt", weights_only=True)
        assert saved["selected_round"] == best["round"]
        assert saved["selected_fixed_gamma"] == best["fixed_gamma"]
        assert saved["mode"] == mode
    assert summary["source_knowledge_vectors"] == 14
    diagnostics = summary["outcomes"]["global"]["training_only_gate_diagnostics"]
    assert diagnostics["users"] == 7
    assert diagnostics["std_across_user_means"] < 1e-6
    assert summary["outcomes"]["user"]["training_only_gate_diagnostics"]["std_across_user_means"] > 1e-5
    assert summary["outcomes"]["item"]["training_only_gate_diagnostics"]["mean_std_across_candidate_items"] > 1e-5
