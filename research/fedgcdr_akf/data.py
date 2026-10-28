"""Streaming CSV ingestion and a common chronological cutoff across domains."""

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from itertools import islice
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(data_config, output_dir):
    cfg = data_config
    target, sources = cfg["target"], cfg["sources"]
    domains = [target, *sources]
    if not sources or len(set(domains)) != len(domains):
        raise ValueError("Choose a target and distinct source domains.")
    if any(Path(name).name != name or name in {".", ".."} for name in domains):
        raise ValueError("Domain names must be CSV basenames.")
    train_fraction = cfg["train_fraction"]
    valid_fraction = cfg["validation_fraction"]
    if not 0 < train_fraction < train_fraction + valid_fraction < 1:
        raise ValueError("Train and validation fractions must leave a nonempty test fraction.")
    limit, max_users = cfg["max_rows_per_domain"], cfg["max_users"]
    if limit < 0 or max_users < 0 or cfg["min_train_interactions"] < 1:
        raise ValueError("Row/user limits must be nonnegative and minimum history positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(output / "interactions.sqlite")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("""CREATE TABLE events (
        domain TEXT, user TEXT, item TEXT, ts INTEGER,
        PRIMARY KEY(domain, user, item)) WITHOUT ROWID""")
    manifest = []
    try:
        for domain in domains:
            path = Path(cfg["ratings_dir"]) / f"{domain}.csv"
            digest, counts, batch = hashlib.sha256(), Counter(), []
            with path.open("rb") as handle:
                for raw in islice(handle, limit or None):
                    digest.update(raw)
                    counts["rows_read"] += 1
                    counts["bytes_read"] += len(raw)
                    row = next(csv.reader([raw.decode("utf-8-sig")]))
                    if len(row) != 4 or not row[0] or not row[1]:
                        raise ValueError(f"Invalid four-column CSV record in {domain}, row {counts['rows_read']}")
                    rating, timestamp = float(row[2]), int(row[3])
                    if not math.isfinite(rating) or not 1 <= rating <= 5 or timestamp < 0:
                        raise ValueError(f"Invalid rating/timestamp in {domain}, row {counts['rows_read']}")
                    if rating >= cfg["rating_threshold"]:
                        batch.append((domain, row[1], row[0], timestamp))
                        counts["positive_rows"] += 1
                    if len(batch) >= 10000:
                        _insert(db, batch)
                        batch.clear()
            _insert(db, batch)
            db.commit()
            unique = db.execute("SELECT COUNT(*) FROM events WHERE domain=?", (domain,)).fetchone()[0]
            entry = {
                "domain": domain, "path": str(path.resolve()),
                "file_size_bytes": path.stat().st_size, **counts,
                "hash_scope": "full_file" if counts["bytes_read"] == path.stat().st_size else "consumed_prefix",
                "sha256": digest.hexdigest(), "unique_positive_pairs": unique,
                "duplicate_positive_rows": counts["positive_rows"] - unique,
            }
            manifest.append(entry)
            print(json.dumps({"ingested": domain, "rows": counts["rows_read"], "positive_pairs": unique}), flush=True)
        db.execute("CREATE INDEX domain_time ON events(domain, ts)")
        size = db.execute("SELECT COUNT(*) FROM events WHERE domain=?", (target,)).fetchone()[0]
        if size < 10:
            raise ValueError("Insufficient target interactions for chronological splitting.")
        cuts = []
        for fraction in (train_fraction, train_fraction + valid_fraction):
            cuts.append(db.execute(
                "SELECT ts FROM events WHERE domain=? ORDER BY ts LIMIT 1 OFFSET ?",
                (target, min(size - 1, int(size * fraction))),
            ).fetchone()[0])
        train_cut, valid_cut = cuts
        if train_cut >= valid_cut:
            raise ValueError("Timestamp ties leave no validation interval; use more data or explicit cutoffs.")
        users = [row[0] for row in db.execute(
            "SELECT user FROM events WHERE domain=? AND ts<=? GROUP BY user HAVING COUNT(*)>=?",
            (target, train_cut, cfg["min_train_interactions"]),
        )]
        seed = cfg["seed"]
        users.sort(key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).digest())
        if max_users:
            users = users[:max_users]
        if not users:
            raise ValueError("No target users satisfy the training-only history requirement.")
        user_ids = {user: uid for uid, user in enumerate(users)}
        db.execute("CREATE TEMP TABLE cohort(user TEXT PRIMARY KEY, uid INTEGER)")
        db.executemany("INSERT INTO cohort VALUES (?,?)", user_ids.items())
        prepared_domains = {}
        summary_domains = {}
        for domain in domains:
            records = list(db.execute(
                "SELECT c.uid,e.item,e.ts FROM events e JOIN cohort c ON c.user=e.user "
                "WHERE e.domain=? AND e.ts<=? ORDER BY e.ts,e.item", (domain, train_cut),
            ))
            item_names = sorted({item for _, item, _ in records})
            if not item_names:
                raise ValueError(f"No cohort training interactions for source {domain}; increase prefix/user limits.")
            item_ids = {name: idx for idx, name in enumerate(item_names)}
            train = [[] for _ in users]
            for uid, item, _ in records:
                train[uid].append(item_ids[item])
            splits, coverage = {}, {}
            if domain == target:
                for name, lower, upper in (("validation", train_cut, valid_cut), ("test", valid_cut, 2**63 - 1)):
                    retained = [[] for _ in users]
                    rows = list(db.execute(
                        "SELECT c.uid,e.item FROM events e JOIN cohort c ON c.user=e.user "
                        "WHERE e.domain=? AND e.ts>? AND e.ts<=? ORDER BY e.ts,e.item",
                        (domain, lower, upper),
                    ))
                    for uid, item in rows:
                        if item in item_ids:
                            retained[uid].append(item_ids[item])
                    kept = sum(map(len, retained))
                    splits[name] = retained
                    coverage[name] = {
                        "cohort_positive_pairs": len(rows), "warm_item_positive_pairs": kept,
                        "dropped_unseen_items": len(rows) - kept,
                        "evaluated_users": sum(bool(items) for items in retained),
                    }
                    if not kept:
                        raise ValueError(f"No warm-item {name} interactions; increase sample/cohort size.")
            prepared_domains[domain] = {"items": item_names, "train": train, **splits}
            summary_domains[domain] = {
                "train_users": sum(bool(items) for items in train), "train_items": len(item_names),
                "train_pairs": len(records), "max_train_timestamp": max(t for _, _, t in records),
                **coverage,
            }
        prepared = {
            "schema_version": 1, "config": cfg, "target": target, "sources": sources,
            "user_ids": users, "train_cutoff": train_cut, "validation_cutoff": valid_cut,
            "domains": prepared_domains,
        }
        summary = {
            "config": cfg, "scope": "bounded_prefix_pilot" if limit or max_users else "full_input_target_cohort",
            "cohort_rule": "Target users selected by training-only history, then seeded ID hash; sources restricted to this cohort.",
            "deduplication": "Earliest positive event per domain/user/item, rating >= threshold.",
            "split_rule": "Target timestamp quantiles; all domain training timestamps <= target train cutoff; warm items only.",
            "train_cutoff": train_cut, "validation_cutoff": valid_cut, "users": len(users),
            "source_overlap_users": sum(any(prepared_domains[s]["train"][u] for s in sources) for u in range(len(users))),
            "domains": summary_domains, "input_manifest": manifest,
            "provenance": "User-supplied CSV files; dataset release and ID semantics not independently verified.",
        }
        write_json(output / "prepared.json", prepared)
        write_json(output / "summary.json", summary)
        return summary
    finally:
        db.close()


def _insert(db, batch):
    db.executemany(
        "INSERT INTO events VALUES (?,?,?,?) ON CONFLICT(domain,user,item) "
        "DO UPDATE SET ts=MIN(ts,excluded.ts)", batch,
    )


def load_prepared(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
