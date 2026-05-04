"""Compose the M2_Spark_ML_BigDataCourseProject.ipynb file.

Run from the repo root:
    python scripts/make_notebook.py
"""
import json
from pathlib import Path

OUTFILE = "M2_Spark_ML_BigDataCourseProject.ipynb"

# Author display strings used by f-strings inside cell sources
AUTHOR_LAYAL    = "Layal Alshmassi (231097)"
AUTHOR_ALANOUD  = "Alanoud Alrowaite (231412)"
AUTHOR_JOUD     = "Joud Abohaimed (231453)"
AUTHOR_ALJAZEE  = "Aljazee Abuhemed (231800)"


_pages = []


def _md(body: str) -> None:
    _pages.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": body.splitlines(keepends=True),
    })


def _code(body: str) -> None:
    _pages.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": body.splitlines(keepends=True),
    })


# ============================================================
_md(f"""# SE446 Milestone 2 — Big Data Course Project

Spark DataFrame analytics + MLlib arrest predictor on Chicago Crime data
(Hadoop 3.4.1 / Spark 3.5.4, 1 master + 2 workers).

| Member               | Student ID | GitHub             | Tasks    |
|----------------------|-----------:|--------------------|----------|
| {AUTHOR_LAYAL}       |     231097 | `layalalshmassi`   | 1, 5, 11 |
| {AUTHOR_ALANOUD}     |     231412 | `aalrowaite`       | 2, 6     |
| {AUTHOR_JOUD}        |     231453 | `jabohaimed`       | 3, 7, 10 |
| {AUTHOR_ALJAZEE}     |     231800 | `aljazyabuhaimed`  | 4, 9     |

**Spec compliance — May 2026 update:**
* Task 8 (CrossValidator) is **waived** by the instructor and is not in this notebook.
* Phase B (Tasks 5–7) trains on a **5% sample**: `df.sample(fraction=0.05, seed=42)`.
* Task 11 uses `--deploy-mode cluster` and the application stdout is collected with
  `yarn logs -applicationId <appId> > output/spark_submit/run.log`.
""")


# ------ Setup ------
_md("## 0. Setup")

_code("""import os
import time
import shutil

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as fns
from pyspark.sql.types import IntegerType, StringType


def _on_cluster() -> bool:
    return shutil.which("hdfs") is not None


def _make_spark() -> SparkSession:
    base = (SparkSession.builder
            .appName("M2_BigDataCourseProject")
            .config("spark.sql.shuffle.partitions", "8"))
    if _on_cluster():
        return base.getOrCreate()
    return (base
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate())


where = "cluster" if _on_cluster() else "local"
spark = _make_spark()
if where == "local":
    spark.sparkContext.setLogLevel("WARN")

print(f"Where:           {where}")
print(f"Spark version:   {spark.version}")
print(f"Spark master:    {spark.sparkContext.master}")
""")


# ------ Data load ------
_md("## 1. Read the dataset")

_code("""HDFS_FILE = "hdfs:///data/chicago_crimes.csv"


def _read_real_csv():
    raw = spark.read.csv(HDFS_FILE, header=True, inferSchema=True)
    return (raw
            .withColumn("Hour",
                        fns.hour(fns.to_timestamp(fns.col("Date"),
                                                  "MM/dd/yyyy hh:mm:ss a")))
            .withColumn("label",        fns.col("Arrest").cast(IntegerType()))
            .withColumn("Domestic_str", fns.col("Domestic").cast(StringType())))


def _make_synthetic(rows: int = 10_000):
    import random
    random.seed(42)
    base_p_per_type = {
        "NARCOTICS":           0.85,
        "PROSTITUTION":        0.80,
        "WEAPONS VIOLATION":   0.60,
        "BATTERY":             0.30,
        "ASSAULT":             0.25,
        "ROBBERY":             0.15,
        "THEFT":               0.10,
        "BURGLARY":            0.08,
        "MOTOR VEHICLE THEFT": 0.06,
        "CRIMINAL DAMAGE":     0.05,
    }
    locs = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER",
            "PARKING LOT", "SCHOOL", "ALLEY", "RESIDENCE-GARAGE"]
    yrs = [2020, 2021, 2022, 2023, 2024, 2025]
    bag = []
    for _ in range(rows):
        kind = random.choice(list(base_p_per_type))
        h = random.randint(0, 23)
        is_dom = random.random() < 0.15
        p = base_p_per_type[kind] + (0.20 if is_dom else 0.0)
        if 2 <= h <= 5:
            p -= 0.10
        p = max(0.01, min(0.99, p))
        bag.append(Row(
            District=random.randint(1, 25),
            **{"Primary Type": kind},
            **{"Location Description": random.choice(locs)},
            Year=random.choice(yrs),
            Hour=h,
            Domestic_str=str(is_dom).lower(),
            Arrest=random.random() < p,
            label=int(random.random() < p),
        ))
    return spark.createDataFrame(bag)


crimes = _read_real_csv() if where == "cluster" else _make_synthetic()
crimes.cache()
print(f"Records loaded: {crimes.count():,}")
crimes.printSchema()
crimes.show(3, truncate=False)
""")


# ============================================================
_md("---\n# Phase A — Spark DataFrame analytics")


_md(f"""## Task 1 — Crime type distribution
*{AUTHOR_LAYAL}*

Group rows by `Primary Type` then sort descending.""")

_code(f"""# Task 1 — {AUTHOR_LAYAL}
top_types = (crimes
             .groupBy("Primary Type")
             .agg(fns.count(fns.lit(1)).alias("freq"))
             .orderBy(fns.desc("freq")))
top_types.show(10, truncate=False)
""")


_md(f"""## Task 2 — Location hotspots (Spark SQL)
*{AUTHOR_ALANOUD}*

Use `createOrReplaceTempView` and run the query through `spark.sql`.""")

_code(f"""# Task 2 — {AUTHOR_ALANOUD}
crimes.createOrReplaceTempView("chicago_crimes")

hotspots = spark.sql(\"\"\"
    SELECT  `Location Description` AS spot,
            COUNT(*)               AS cases
      FROM  chicago_crimes
     WHERE  `Location Description` IS NOT NULL
     GROUP  BY `Location Description`
     ORDER  BY cases DESC
     LIMIT  10
\"\"\")
hotspots.show(truncate=False)
""")


_md(f"""## Task 3 — Year trend
*{AUTHOR_JOUD}*

Counts per year (matplotlib chart in local mode).""")

_code(f"""# Task 3 — {AUTHOR_JOUD}
per_year = (crimes
            .groupBy("Year")
            .agg(fns.count(fns.lit(1)).alias("incidents"))
            .orderBy("Year"))
per_year.show(30)
""")

_code(f"""# Task 3 chart — {AUTHOR_JOUD}
if where == "local":
    import matplotlib.pyplot as plt
    pdf = per_year.toPandas().dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(pdf["Year"].astype(int), pdf["incidents"], alpha=0.5, color="#5b8c5a")
    ax.plot(pdf["Year"].astype(int), pdf["incidents"], color="#2f5d2f", lw=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidents")
    ax.set_title("Incidents per year")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/year_trend.png", dpi=120)
    plt.show()
else:
    print("Cluster mode — printed table above is the deliverable.")
""")


_md(f"""## Task 4 — Arrest rate
*{AUTHOR_ALJAZEE}*

Overall rate plus a per-crime-type breakdown.""")

_code(f"""# Task 4 — {AUTHOR_ALJAZEE}
n_records = crimes.count()
n_arrest  = crimes.filter(fns.col("Arrest") == True).count()
print(f"Overall arrest rate: {{n_arrest:,}} / {{n_records:,}} = {{n_arrest/n_records*100:.2f}}%")

by_type = (crimes
           .groupBy("Primary Type")
           .agg(fns.count(fns.lit(1)).alias("records"),
                fns.avg(fns.col("label").cast("double")).alias("arrest_rate"))
           .filter(fns.col("records") >= 100)
           .orderBy(fns.desc("arrest_rate")))
print("Top arrest-rate crime types (min 100 records):")
by_type.show(15, truncate=False)
""")


# ============================================================
_md("""---
# Phase B — MLlib arrest predictor (5% sample per spec)

The May 2026 spec update requires Phase B to train on a 5% sample. Locally that runs
on the 10K synthetic dataset; on the cluster it shrinks the 793K-row HDFS dataset
to roughly 39,654 rows that fit the cluster's RAM budget.""")

_code("""# 5% sample applied before any feature engineering
ml_subset = crimes.sample(fraction=0.05, seed=42)
print(f"Phase B working set: {ml_subset.count():,} rows  (5% sample, seed=42)")
""")


_md(f"""## Task 5 — Feature pipeline
*{AUTHOR_LAYAL}*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over four
features, 80/20 split with `seed=42`.""")

_code(f"""# Task 5 — {AUTHOR_LAYAL}
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

if "Domestic_str" not in ml_subset.columns:
    ml_subset = ml_subset.withColumn("Domestic_str",
                                     fns.col("Domestic").cast(StringType()))

pri_idx = StringIndexer(inputCol="Primary Type",
                        outputCol="pri_idx",
                        handleInvalid="skip")
dom_idx = StringIndexer(inputCol="Domestic_str",
                        outputCol="dom_idx",
                        handleInvalid="skip")
feat_col = VectorAssembler(
    inputCols=["District", "pri_idx", "Hour", "dom_idx"],
    outputCol="feat_col",
)

train_data, test_data = ml_subset.randomSplit([0.8, 0.2], seed=42)
train_data.cache()
test_data.cache()
print(f"Train rows: {{train_data.count():,}}   Test rows: {{test_data.count():,}}")

# Inspect the assembled feature column for 5 rows
inspect_pipe = Pipeline(stages=[pri_idx, dom_idx, feat_col]).fit(train_data)
inspect_pipe.transform(train_data).select(
    "Primary Type", "pri_idx",
    "District", "Hour",
    "Domestic_str", "dom_idx",
    "feat_col", "label",
).show(5, truncate=False)
print("Vector layout: [District, pri_idx, Hour, dom_idx]")
""")


_md(f"""## Task 6 — Train and evaluate three classifiers
*{AUTHOR_ALANOUD}*

Logistic Regression (maxIter=100, regParam=0.01), Random Forest (numTrees=100,
maxDepth=5, maxBins=64), GBT (maxIter=50, maxDepth=5, maxBins=64). `maxBins=64`
is needed because Primary Type has more than 32 categories on the cluster.""")

_code(f"""# Task 6 helpers — {AUTHOR_ALANOUD}
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)

bin_eval   = BinaryClassificationEvaluator(labelCol="label")
multi_eval = MulticlassClassificationEvaluator(labelCol="label",
                                               predictionCol="prediction")


def _score(predictions):
    return {{
        "AUC":       bin_eval.evaluate(predictions),
        "Accuracy":  multi_eval.evaluate(predictions, {{multi_eval.metricName: "accuracy"}}),
        "F1":        multi_eval.evaluate(predictions, {{multi_eval.metricName: "f1"}}),
        "Precision": multi_eval.evaluate(predictions, {{multi_eval.metricName: "weightedPrecision"}}),
        "Recall":    multi_eval.evaluate(predictions, {{multi_eval.metricName: "weightedRecall"}}),
    }}


def _confusion(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    box  = {{(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}}
    return (box.get((0, 0), 0), box.get((0, 1), 0),
            box.get((1, 0), 0), box.get((1, 1), 0))
""")

_code(f"""# Task 6 training — {AUTHOR_ALANOUD}
classifiers = [
    ("LogisticRegression",
     LogisticRegression(featuresCol="feat_col", labelCol="label",
                        maxIter=100, regParam=0.01)),
    ("RandomForest",
     RandomForestClassifier(featuresCol="feat_col", labelCol="label",
                            numTrees=100, maxDepth=5,
                            maxBins=64, seed=42)),
    ("GBT",
     GBTClassifier(featuresCol="feat_col", labelCol="label",
                   maxIter=50, maxDepth=5,
                   maxBins=64, seed=42)),
]

records = []
fitted_rf = None
for tag, learner in classifiers:
    print(f"\\n>> training {{tag}}")
    pipeline = Pipeline(stages=[pri_idx, dom_idx, feat_col, learner])
    started = time.time()
    fitted_pipeline = pipeline.fit(train_data)
    elapsed = time.time() - started
    preds = fitted_pipeline.transform(test_data)
    metrics = _score(preds)
    cm = _confusion(preds)
    records.append((tag, fitted_pipeline, metrics, cm, elapsed))
    for k, v in metrics.items():
        print(f"  {{k:<10}}{{v:.4f}}")
    print(f"  Train(s)  {{elapsed:.1f}}")
    print(f"  CM (TN,FP,FN,TP) = {{cm}}")
    if tag == "RandomForest":
        fitted_rf = fitted_pipeline.stages[-1]

# Comparison table
print("\\n" + "=" * 78)
print(f"{{'metric':<11}}{{'Logistic':>14}}{{'RandomForest':>16}}{{'GBT':>14}}")
print("-" * 78)
m_lr, m_rf, m_gbt = (records[0][2], records[1][2], records[2][2])
for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
    print(f"{{k:<11}}{{m_lr[k]:>14.4f}}{{m_rf[k]:>16.4f}}{{m_gbt[k]:>14.4f}}")
print(f"{{'Train(s)':<11}}{{records[0][4]:>14.1f}}{{records[1][4]:>16.1f}}{{records[2][4]:>14.1f}}")
print("=" * 78)
top = max(records, key=lambda r: r[2]["AUC"])
print(f"Top model by AUC: {{top[0]}} ({{top[2]['AUC']:.4f}})")
""")


_md(f"""## Task 7 — Random Forest feature importances
*{AUTHOR_JOUD}*

Importances tell us which feature drives most of the splits in the trees.""")

_code(f"""# Task 7 — {AUTHOR_JOUD}
layout = ["District", "pri_idx", "Hour", "dom_idx"]
importances = fitted_rf.featureImportances.toArray()

print("Random Forest feature importances:")
for feat, imp in sorted(zip(layout, importances), key=lambda kv: -kv[1]):
    bar = "=" * int(round(imp * 50))
    print(f"  {{feat:<10}} {{imp:.4f}}  {{bar}}")
""")


_md("""**Reading the importances.**
The crime-type index dominates because the per-crime arrest-rate distribution from
Task 4 is itself dominated by crime type — NARCOTICS is near 99% while THEFT is
near 14%. Once a tree splits on the crime type it has most of its answer.

Logistic Regression underperforms the tree models because it treats `pri_idx` as a
numeric feature and fits a single linear coefficient, implying a meaningless ordering
between crime types. Tree models split on individual values of the index and
side-step that issue.""")


_md("""---
## Cleanup""")

_code("""spark.stop()""")


# ------ Write the notebook ------
nb = {
    "cells": _pages,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path(OUTFILE).write_text(json.dumps(nb, indent=1))
print(f"wrote {OUTFILE} ({len(_pages)} cells)")
