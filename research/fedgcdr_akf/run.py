"""Reproducible pretraining, fusion comparisons and validation-only selection."""

import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import load_prepared, write_json
from .federated import evaluate, extract_knowledge, gate_diagnostics, initialize_domain, train_round
from .model import MODES, FusionHead
from .upstream import MODEL_PATH, ROOT, UPSTREAM_COMMIT


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(config, prepared_path, output_dir):
    cfg = config["training"]
    data = load_prepared(prepared_path)
    if data["config"] != config["data"]:
        raise ValueError("Prepared data configuration differs from this run's data configuration.")
    if not cfg["modes"] or len(set(cfg["modes"])) != len(cfg["modes"]) or set(cfg["modes"]) - set(MODES):
        raise ValueError("Specify distinct supported comparison modes.")
    for key in ("threads", "embedding_dim", "pretrain_rounds", "fusion_rounds", "local_epochs", "client_batch_size", "max_history", "max_positives"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive.")
    if cfg["embedding_dim"] < 2 or cfg["negatives_per_positive"] < 1 or cfg["learning_rate"] <= 0:
        raise ValueError("Invalid embedding size, negative sampling count or learning rate.")
    gamma_candidates = cfg.get("fixed_gamma_candidates", [cfg["fixed_gamma"]])
    if not gamma_candidates or len(set(gamma_candidates)) != len(gamma_candidates) or any(not 0 <= gamma <= 1 for gamma in gamma_candidates):
        raise ValueError("Fixed gamma candidates must be distinct values in [0, 1].")
    revision = subprocess.check_output(["git", "-C", str(ROOT / "upstream/FedGCDR"), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("Upstream checkout does not match the pinned revision.")
    expected = json.loads((ROOT / "upstream/source_manifest.json").read_text())
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != expected["model_sha256"]:
        raise ValueError("Upstream model differs from the recorded source hash.")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is not available.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    seed_everything(cfg["seed"])
    num_users = len(data["user_ids"])
    source_names, target = data["sources"], data["target"]
    domain_names = [*source_names, target]
    states, pretrain_logs = {}, []
    start = time.perf_counter()
    log = (output / "history.jsonl").open("w", encoding="utf-8")

    def record(event):
        log.write(json.dumps(event, ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps(event, ensure_ascii=False), flush=True)

    try:
        for domain_index, name in enumerate(domain_names):
            domain = data["domains"][name]
            state = initialize_domain(num_users, len(domain["items"]), cfg["embedding_dim"], device)
            rng = random.Random(cfg["seed"] + domain_index)
            for round_id in range(cfg["pretrain_rounds"]):
                metrics = train_round(state.gat, state.users, None, state.items, domain["train"], cfg, rng)
                event = {"stage": "pretrain", "domain": name, "round": round_id + 1, **metrics}
                pretrain_logs.append(event)
                record(event)
            states[name] = state
        extracted = [extract_knowledge(states[name], data["domains"][name]["train"], cfg["max_history"]) for name in source_names]
        knowledge = torch.stack([value for value, _ in extracted], dim=1).detach()
        available = torch.stack([mask for _, mask in extracted], dim=1)
        knowledge_bytes = int(available.sum()) * cfg["embedding_dim"] * knowledge.element_size()
        base = states[target]
        train = data["domains"][target]["train"]
        validation = data["domains"][target]["validation"]
        test = data["domains"][target]["test"]
        torch.save({"knowledge": knowledge.cpu(), "available": available.cpu(),
                    "target_users": base.users.cpu(), "target_items": base.items.cpu(),
                    "target_gat": base.gat.state_dict()}, output / "pretrained.pt")
        outcomes, user_test_rows = {}, {}
        for mode in cfg["modes"]:
            best_key, best, best_metrics, best_round = None, None, None, None
            total_payload, trials = 0, []
            mode_start = time.perf_counter()
            candidates = gamma_candidates if mode in {"fixed", "fixed_capacity"} else [cfg["fixed_gamma"]]
            for gamma in candidates:
                # Every candidate sees the same initialization and training samples.
                seed_everything(cfg["seed"] + 1000)
                head = FusionHead(cfg["embedding_dim"], len(source_names), mode, gamma, device)
                head.gat.load_state_dict(base.gat.state_dict())
                users, graph_users, items = base.users.clone(), base.users.clone(), base.items.clone()
                rng = random.Random(cfg["seed"] + 2000)
                trial_key, trial_metrics, trial_round, trial_payload = None, None, None, 0
                for round_id in range(cfg["fusion_rounds"]):
                    head.train()
                    trained = train_round(head, users, graph_users, items, train, cfg, rng, knowledge, available)
                    trial_payload += trained["payload_bytes_estimate"]
                    val_metrics, _ = evaluate(head, users, graph_users, items, train, validation, knowledge, available, cfg["max_history"])
                    record({"stage": "fusion", "split": "validation", "mode": mode, "fixed_gamma": gamma,
                            "round": round_id + 1, **trained, "metrics": val_metrics})
                    key = (val_metrics["ndcg@10"], val_metrics["recall@10"])
                    if trial_key is None or key > trial_key:
                        trial_key, trial_metrics, trial_round = key, val_metrics, round_id + 1
                    if best_key is None or key > best_key:
                        best_key, best_round, best_metrics = key, round_id + 1, val_metrics
                        best = {"head": copy.deepcopy(head.state_dict()), "users": users.clone(),
                                "graph_users": graph_users.clone(), "items": items.clone(), "fixed_gamma": gamma}
                trials.append({"fixed_gamma": gamma, "selected_round": trial_round,
                               "validation": trial_metrics, "payload_bytes_estimate": trial_payload})
                total_payload += trial_payload
            # The test labels have not been accessed by any gamma candidate.
            head = FusionHead(cfg["embedding_dim"], len(source_names), mode, best["fixed_gamma"], device)
            head.load_state_dict(best["head"])
            test_metrics, rows = evaluate(head, best["users"], best["graph_users"], best["items"],
                                         train, test, knowledge, available, cfg["max_history"])
            record({"stage": "final", "split": "test", "mode": mode, "selected_round": best_round,
                    "selected_fixed_gamma": best["fixed_gamma"], "metrics": test_metrics})
            checkpoint = {"mode": mode, "selected_round": best_round, "validation_metrics": best_metrics,
                          "selected_fixed_gamma": best["fixed_gamma"],
                          "config": config, "upstream_commit": revision,
                          "head": {key: value.cpu() for key, value in best["head"].items()},
                          **{key: best[key].cpu() for key in ("users", "graph_users", "items")}}
            torch.save(checkpoint, output / f"{mode}.pt")
            write_json(output / f"{mode}_test_users.json", rows)
            outcome = {
                "selected_round": best_round, "validation": best_metrics, "test": test_metrics,
                "selected_fixed_gamma": best["fixed_gamma"] if mode in {"fixed", "fixed_capacity"} else None,
                "validation_trials": trials,
                "shared_module_parameters": sum(p.numel() for p in head.parameters()),
                "active_shared_parameters": sum(p.numel() for p in head.active_parameters()),
                "embedding_parameters": best["users"].numel() + best["graph_users"].numel() + best["items"].numel(),
                "training_only_gate_diagnostics": gate_diagnostics(head, best["users"], best["graph_users"], best["items"],
                                                                   train, knowledge, available, cfg["max_history"]),
                "fusion_payload_bytes_estimate": next(trial["payload_bytes_estimate"] for trial in trials if trial["fixed_gamma"] == best["fixed_gamma"]),
                "total_search_payload_bytes_estimate": total_payload,
                "elapsed_seconds": time.perf_counter() - mode_start,
            }
            outcomes[mode], user_test_rows[mode] = outcome, rows
        for reference in ("local", "fixed"):
            if reference not in outcomes:
                continue
            reference_rows = {row["user_index"]: row for row in user_test_rows[reference]}
            for mode, rows in user_test_rows.items():
                changes = [row["ndcg@10"] - reference_rows[row["user_index"]]["ndcg@10"] for row in rows]
                outcomes[mode][f"test_ndcg_change_vs_{reference}"] = {
                    "mean": sum(changes) / len(changes),
                    "fraction_decreased": sum(change < 0 for change in changes) / len(changes),
                    "fraction_unchanged": sum(change == 0 for change in changes) / len(changes),
                }
        summary = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "implementation_smoke_only" if data["config"]["max_rows_per_domain"] or data["config"]["max_users"] else "research_run",
            "upstream": expected, "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
            "config": config, "prepared_sha256": hashlib.sha256(Path(prepared_path).read_bytes()).hexdigest(),
            "data_summary": load_prepared(Path(prepared_path).with_name("summary.json")),
            "source_knowledge_vectors": int(available.sum()), "users_with_source_knowledge": int(available.any(dim=1).sum()),
            "source_knowledge_payload_bytes_estimate": knowledge_bytes,
            "pretrain_payload_bytes_estimate": sum(row["payload_bytes_estimate"] for row in pretrain_logs),
            "selection": "Best validation NDCG@10, then Recall@10; first tied candidate/round; test once per declared mode after all gamma trials.",
            "candidate_protocol": "All cohort training items, excluding each user's training positives; warm user/item evaluation.",
            "limitations": ["In-process federated simulation, no formal DP/security aggregation.",
                            "Upstream GAT/MLP reused; graph_transfer is a protocol adaptation, not an exact paper reproduction.",
                            "Bounded prefix and cohort runs do not support recommendation-effectiveness claims.",
                            "fixed_capacity is approximate parameter matching; formal runs still require validation-only hyperparameter search and multiple seeds."],
            "elapsed_seconds": time.perf_counter() - start, "outcomes": outcomes,
        }
        write_json(output / "summary.json", summary)
        return summary
    finally:
        log.close()
