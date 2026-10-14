"""Stage eval.baselines_val (CLAUDE.md s8-9, Week 4): fit every baseline on train and evaluate it
on validation, with strata, bootstrap intervals, paired comparisons, tables and figures.

Baselines are cheap to fit, so they are refit here instead of keeping model artifacts (D40);
LightGCN and the projection hybrid (Week 5) keep artifacts with manifests.
"""

import json

import numpy as np
import pandas as pd

from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage
from ml.eval import report
from ml.eval.data import build_eval_data
from ml.eval.evaluator import EvalResult, evaluate
from ml.eval.strata import by_sparsity, cold_item_rows, longtail_rows
from ml.models.baselines import (
    ALSScorer,
    ContentScorer,
    ItemKNNScorer,
    PopularityScorer,
    RandomScorer,
)

REFERENCE = "popularity"
PAIRED_METRICS = ("ndcg@10", "recall@20")


@stage(
    "eval.baselines_val",
    inputs=[
        "data:interactions.parquet",
        "data:splits.json",
        "data:id_map.json",
        "artifacts:content/content_item.npy",
    ],
    outputs=["results:val/"],
    config_keys=[
        "eval",
        "data.relevance_min_rating",
        "data.longtail_fraction",
        "data.sparsity_buckets",
        "models.popularity",
        "models.itemknn",
        "models.als",
    ],
)
def build_baselines_val(ctx: StageContext) -> None:
    """Validation results for random, popularity, ItemKNN, ALS and content-only."""
    out = ctx.out("results:val/")
    ids = load_id_map(ctx.path("data:id_map.json"))
    splits = json.loads(ctx.path("data:splits.json").read_text(encoding="utf-8"))
    relevance = float(ctx.param("data.relevance_min_rating"))
    data = build_eval_data(
        pd.read_parquet(ctx.path("data:interactions.parquet")),
        splits,
        "val",
        ids.n_users,
        ids.n_items,
        relevance,
        float(ctx.param("data.longtail_fraction")),
    )
    ctx.cutoff = data.cutoff
    emb = np.load(ctx.path("artifacts:content/content_item.npy"))
    ev = ctx.param("eval")
    ks, n_boot = tuple(int(k) for k in ev["ks"]), int(ev["bootstrap_resamples"])
    buckets = ctx.param("data.sparsity_buckets")
    scorers = [
        RandomScorer(ctx.seed),
        PopularityScorer(ctx.param("models.popularity")),
        ItemKNNScorer(ctx.param("models.itemknn")),
        ALSScorer(ctx.param("models.als"), ctx.seed),
        ContentScorer(emb, relevance),
    ]

    results: dict[str, EvalResult] = {}
    tail_rows, cold_rows, strata = {}, {}, []
    for scorer in scorers:
        scorer.fit(data.fit)
        result = evaluate(scorer, data, ks, embeddings=emb)
        results[scorer.name] = result
        tail_rows[scorer.name] = longtail_rows(result, data, ks)
        cold_rows[scorer.name] = cold_item_rows(result, data, ks)
        strata.append(by_sparsity(result, data, buckets).assign(model=scorer.name))
        result.rows.to_parquet(out / f"{scorer.name}.users.parquet", index=False)
        summary = {
            "model": scorer.name,
            "mode": "val",
            "cutoff": data.cutoff,
            "users": len(result.rows),
            "means": result.means,
            "beyond": result.beyond,
            "longtail_means": tail_rows[scorer.name].drop(columns="u").mean().to_dict(),
            "cold_item_means": cold_rows[scorer.name].drop(columns="u").mean().to_dict()
            if len(cold_rows[scorer.name])
            else {},
        }
        (out / f"{scorer.name}.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
        ctx.log.info("%s: ndcg@10 %.4f", scorer.name, result.means.get("ndcg@10", float("nan")))

    rows = {n: r.rows for n, r in results.items()}
    beyond = {n: r.beyond for n, r in results.items()}
    main = report.main_table(rows, beyond, n_boot, ctx.seed)
    main.to_csv(out / "table_main.csv", index=False)
    strata_table = pd.concat(strata, ignore_index=True)
    strata_table.to_csv(out / "table_strata.csv", index=False)
    longtail = report.main_table(tail_rows, {n: {} for n in tail_rows}, n_boot, ctx.seed).assign(
        stratum="longtail"
    )
    cold = report.main_table(
        {n: r for n, r in cold_rows.items() if len(r)}, {n: {} for n in cold_rows}, n_boot, ctx.seed
    ).assign(stratum="cold_items")
    pd.concat([longtail, cold], ignore_index=True).to_csv(out / "table_longtail.csv", index=False)
    paired = pd.concat(
        [
            report.paired_table(rows, REFERENCE, PAIRED_METRICS, n_boot, ctx.seed),
            report.paired_table(tail_rows, REFERENCE, PAIRED_METRICS, n_boot, ctx.seed, "longtail"),
        ],
        ignore_index=True,
    )
    paired.to_csv(out / "bootstrap.csv", index=False)
    report.plot_bars(main, out / "bars_ndcg_recall.png")
    report.plot_by_bucket(strata_table, out / "ndcg_by_bucket.png")

    ctx.count("evaluated_users", len(data.users))
    ctx.count("catalogue_items", ids.n_items)
    ctx.notes["ndcg@10"] = {n: r.means.get("ndcg@10") for n, r in results.items()}
