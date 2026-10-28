"""Aggregate prespecified seeds without selecting or hiding test outcomes."""

import csv
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .data import load_prepared, write_json
from .federated import ranking_metrics

LABELS = {
    "local": "目标域本地模型", "graph_transfer": "图迁移适配对照", "fixed": "固定 γ（验证集选择）",
    "fixed_capacity": "固定 γ（容量对照）", "global": "全局可学习 γ", "user": "用户级动态 γ",
    "item": "物品条件动态 γ",
}


def moments(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0., "values": values}


def row_metrics(rows):
    return {"users": len(rows), **{key: statistics.mean(row[key] for row in rows) if rows else None
                                  for key in ("recall@10", "ndcg@10")}}


def summarize(output_dir, prepared_path, report_prefix):
    output, prefix = Path(output_dir), Path(report_prefix)
    config = load_prepared(output / "comparison_config.json")
    seeds, modes = config["comparison"]["seeds"], config["training"]["modes"]
    prepared = load_prepared(prepared_path)
    data_summary = load_prepared(Path(prepared_path).with_name("summary.json"))
    summaries, rows_by_seed, audit = {}, {}, {}
    prepared_hash = hashlib.sha256(Path(prepared_path).read_bytes()).hexdigest()
    for seed in seeds:
        run_dir = output / f"seed_{seed}"
        summary = load_prepared(run_dir / "summary.json")
        if summary["prepared_sha256"] != prepared_hash or summary["config"]["training"]["seed"] != seed:
            raise ValueError("Runs must use the same prepared data and their declared training seed.")
        summaries[seed] = summary
        history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
        rows_by_seed[seed], audit[str(seed)] = {}, {}
        for mode in modes:
            validation = [row for row in history if row.get("mode") == mode and row.get("split") == "validation"]
            tests = [row for row in history if row.get("mode") == mode and row.get("split") == "test"]
            trials = len(config["training"].get("fixed_gamma_candidates", [config["training"]["fixed_gamma"]])) if mode in {"fixed", "fixed_capacity"} else 1
            if len(tests) != 1 or len(validation) != trials * config["training"]["fusion_rounds"]:
                raise ValueError(f"Incomplete comparison or unexpected test count: seed={seed}, mode={mode}")
            best = max(validation, key=lambda row: (row["metrics"]["ndcg@10"], row["metrics"]["recall@10"]))
            if tests[0]["selected_round"] != best["round"] or tests[0]["selected_fixed_gamma"] != best["fixed_gamma"]:
                raise ValueError("Selected model disagrees with validation-only selection.")
            if history.index(tests[0]) <= max(history.index(row) for row in validation):
                raise ValueError("Test evaluation preceded the end of validation search.")
            audit[str(seed)][mode] = {"validation_evaluations": len(validation), "final_test_evaluations": 1}
            rows_by_seed[seed][mode] = load_prepared(run_dir / f"{mode}_test_users.json")
    reference_ids = [row["user_index"] for row in rows_by_seed[seeds[0]][modes[0]]]
    for by_mode in rows_by_seed.values():
        for rows in by_mode.values():
            if [row["user_index"] for row in rows] != reference_ids:
                raise ValueError("Paired comparison requires identical evaluated users across all runs.")
    methods = {}
    for mode in modes:
        outcome = {
            key: moments([summaries[seed]["outcomes"][mode]["test"][key] for seed in seeds])
            for key in ("recall@10", "ndcg@10")
        }
        for group, has_source in (("with_source", True), ("without_source", False)):
            grouped = [row_metrics([row for row in rows_by_seed[seed][mode] if row["has_source_knowledge"] == has_source]) for seed in seeds]
            outcome[group] = {"users": grouped[0]["users"], **{
                key: moments([row[key] for row in grouped]) if grouped[0]["users"] else None
                for key in ("recall@10", "ndcg@10")}}
        outcome["selected_rounds"] = [summaries[seed]["outcomes"][mode]["selected_round"] for seed in seeds]
        outcome["selected_fixed_gammas"] = [summaries[seed]["outcomes"][mode]["selected_fixed_gamma"] for seed in seeds]
        outcome["elapsed_seconds_including_search"] = moments([summaries[seed]["outcomes"][mode]["elapsed_seconds"] for seed in seeds])
        outcome["active_shared_parameters"] = summaries[seeds[0]]["outcomes"][mode]["active_shared_parameters"]
        outcome["selected_trial_payload_bytes_estimate"] = moments([summaries[seed]["outcomes"][mode]["fusion_payload_bytes_estimate"] for seed in seeds])
        outcome["training_only_gate_diagnostics_by_seed"] = {str(seed): summaries[seed]["outcomes"][mode]["training_only_gate_diagnostics"] for seed in seeds}
        outcome["by_train_count"] = {}
        for group in ("1_to_4", "5_to_9", "10_plus"):
            rows = [summaries[seed]["outcomes"][mode]["test"]["by_train_count"][group] for seed in seeds]
            outcome["by_train_count"][group] = {"users": rows[0]["users"], **{
                key: moments([row[key] for row in rows]) if rows[0]["users"] else None
                for key in ("recall@10", "ndcg@10")}}
        if "fixed" in modes:
            outcome["paired_difference_vs_fixed"] = {
                key: moments([summaries[seed]["outcomes"][mode]["test"][key] -
                              summaries[seed]["outcomes"]["fixed"]["test"][key] for seed in seeds])
                for key in ("recall@10", "ndcg@10")}
            outcome["fraction_users_with_lower_ndcg_than_fixed"] = moments([
                summaries[seed]["outcomes"][mode]["test_ndcg_change_vs_fixed"]["fraction_decreased"]
                for seed in seeds])
        methods[mode] = outcome
    # A deterministic baseline uses only training interaction frequencies.
    target = prepared["domains"][prepared["target"]]
    frequency = Counter(item for history in target["train"] for item in history)
    order = sorted(range(len(target["items"])), key=lambda item: (-frequency[item], item))
    popularity_rows = []
    for uid, relevant in enumerate(target["test"]):
        if not relevant:
            continue
        seen = set(target["train"][uid])
        ranked = [item for item in order if item not in seen][:10]
        recall, ndcg = ranking_metrics(ranked, relevant)
        popularity_rows.append({"recall@10": recall, "ndcg@10": ndcg,
                                "has_source_knowledge": any(prepared["domains"][source]["train"][uid] for source in prepared["sources"])})
    popularity = row_metrics(popularity_rows)
    popularity["with_source"] = row_metrics([row for row in popularity_rows if row["has_source_knowledge"]])
    root = Path(__file__).resolve().parents[2]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted((root / "research/fedgcdr_akf").glob("*.py"))}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "completed_exploratory_local_comparison",
        "seeds": seeds, "config": config, "prepared_sha256": prepared_hash, "source_sha256": source_hashes,
        "environment": summaries[seeds[0]]["environment"], "upstream": summaries[seeds[0]]["upstream"],
        "data_summary": data_summary, "test_users": len(reference_ids),
        "protocol_audit": audit, "methods": methods, "training_popularity_baseline": popularity,
        "seed_run_elapsed_seconds": {str(seed): summaries[seed]["elapsed_seconds"] for seed in seeds},
        "limitations": ["First-million-row prefixes and a training-selected cohort; not the full dataset.",
                        "Warm-item filtering and missing source knowledge affect the evaluated population.",
                        "Three seeds show stochastic variation but do not establish a general or statistically significant advantage.",
                        "Fixed/capacity-fixed modes each use three gamma trials; other modes use one, so total search times differ.",
                        "graph_transfer adapts upstream modules and is not an exact paper reproduction."]}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(prefix.with_suffix(".json"), result)
    columns = ["mode", "recall_mean", "recall_std", "ndcg_mean", "ndcg_std", "with_source_ndcg_mean", "with_source_ndcg_std"]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for mode, metrics in methods.items():
            source = metrics["with_source"]["ndcg@10"]
            writer.writerow({"mode": mode, "recall_mean": metrics["recall@10"]["mean"], "recall_std": metrics["recall@10"]["std"],
                             "ndcg_mean": metrics["ndcg@10"]["mean"], "ndcg_std": metrics["ndcg@10"]["std"],
                             "with_source_ndcg_mean": source["mean"] if source else None,
                             "with_source_ndcg_std": source["std"] if source else None})
    text = _markdown(result, output)
    prefix.with_suffix(".md").write_text(text, encoding="utf-8")
    return result


def _markdown(result, output):
    data, config, methods = result["data_summary"], result["config"], result["methods"]
    target = data["domains"][config["data"]["target"]]
    source_users = next(iter(methods.values()))["with_source"]["users"]
    training, data_config = config["training"], config["data"]
    date = datetime.fromisoformat(result["created_at_utc"]).astimezone(timezone(timedelta(hours=8))).date()
    row_limit = data_config["max_rows_per_domain"]
    input_scope = f"每域前 {row_limit:,} 行" if row_limit else "各域完整输入"
    seed_names = "、".join(map(str, result["seeds"]))
    gamma_names = "、".join(map(str, training.get("fixed_gamma_candidates", [training["fixed_gamma"]])))
    lines = [
        "# 本地小规模对比结果", "", f"试验日期：{date}（北京时间）。所有预先声明的方法和种子均保留。", "",
        f"{data_config['target']} 为目标域，{'、'.join(data_config['sources'])} 为源域；{input_scope}。"
        f"训练用户 {data['users']:,} 人，其中 {data['source_overlap_users']:,} 人有源知识；"
        f"目标训练交互 {target['train_pairs']:,} 条、物品 {target['train_items']:,} 个。",
        f"验证用户 {target['validation']['evaluated_users']} 人；测试用户 {result['test_users']} 人，"
        f"其中 {source_users} 人有源知识。目标验证/测试分别因未见物品丢弃 "
        f"{target['validation']['dropped_unseen_items']}/{target['test']['dropped_unseen_items']} 条交互。", "",
        f"使用同一时间划分，训练种子为 {seed_names}。{training['embedding_dim']} 维表示，"
        f"各域预训练 {training['pretrain_rounds']} 轮、目标融合 {training['fusion_rounds']} 轮。"
        f"固定 γ 与容量对照分别在 {gamma_names} 中按验证集选择；每方法每种子只测试一次选定检查点。"
        "排序候选为完整目标训练物品表，屏蔽用户训练正反馈。", "",
        "## 最终测试", "", "数值为三种子均值 ± 样本标准差；有源知识分组使用同一组测试用户。", "",
        "| 方法 | Recall@10 | NDCG@10 | 有源知识用户 NDCG@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    def formatted(value):
        return f"{value['mean']:.5f} ± {value['std']:.5f}" if value else "—"
    for mode, metrics in methods.items():
        lines.append(f"| {LABELS[mode]} | {formatted(metrics['recall@10'])} | {formatted(metrics['ndcg@10'])} | {formatted(metrics['with_source']['ndcg@10'])} |")
    pop = result["training_popularity_baseline"]
    lines.extend([f"| 训练流行度（确定性） | {pop['recall@10']:.5f} | {pop['ndcg@10']:.5f} | {pop['with_source']['ndcg@10']:.5f} |",
                  "", "## 动态权重相对固定权重", ""])
    for mode in ("user", "item"):
        if mode in methods:
            difference = methods[mode]["paired_difference_vs_fixed"]["ndcg@10"]
            lines.append(f"- {LABELS[mode]}：NDCG@10 配对种子差值均值 {difference['mean']:+.6f}，"
                         f"三个差值为 {', '.join(f'{value:+.6f}' for value in difference['values'])}；"
                         f"平均有 {methods[mode]['fraction_users_with_lower_ndcg_than_fixed']['mean']:.1%} 的测试用户 NDCG 下降。")
    lines.extend(["", "## 训练数据上的门控诊断", "",
                  "仅统计有源知识的训练用户，以固定的训练物品探针检查条件变化；以下为三个种子的平均诊断值。", "",
                  "| 模式 | γ 均值 | 用户间 γ 均值标准差 | 同用户候选物品间 γ 标准差 |",
                  "| --- | ---: | ---: | ---: |"])
    for mode in ("fixed", "global", "user", "item"):
        if mode not in methods:
            continue
        diagnostics = list(methods[mode]["training_only_gate_diagnostics_by_seed"].values())
        values = [statistics.mean(row[key] for row in diagnostics)
                  for key in ("mean", "std_across_user_means", "mean_std_across_candidate_items")]
        lines.append(f"| {LABELS[mode]} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    lines.extend(["", "## 选模与预算", "", "| 方法 | 三个种子的选定轮次 | 三个种子的固定 γ | 每种子耗时均值（含搜索，秒） | 活跃共享参数 |",
                  "| --- | --- | --- | ---: | ---: |"])
    for mode, metrics in methods.items():
        gamma = ", ".join(str(value) for value in metrics["selected_fixed_gammas"] if value is not None) or "—"
        lines.append(f"| {LABELS[mode]} | {', '.join(map(str,metrics['selected_rounds']))} | {gamma} | "
                     f"{metrics['elapsed_seconds_including_search']['mean']:.1f} | {metrics['active_shared_parameters']} |")
    lines.extend(["", "固定权重两组各有三个候选，其余方法各有一个候选，耗时包含全部搜索，不能直接作为等调参预算的效率排名。", "",
                  "## 解释范围", "", "这是文件头子集上的探索性比较，仍存在物品顺序、暖用户/暖物品筛选偏差。"
                  "三个随机种子和门控发生变化均不足以证明普遍收益；应同时检查总体、有源知识分组和流行度参考。"
                  "图迁移对照属于上游模块的协议适配；本轮为单机联邦模拟，尚无正式隐私保证。", "",
                  f"日志及检查点：`{output}`。机器可读聚合结果与 CSV 和本报告同名；"
                  "其中保留逐种子指标、分组、门控诊断、来源哈希及验证/测试次数审计。", ""])
    return "\n".join(lines)
