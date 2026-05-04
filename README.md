# SE446 Milestone 2 — Big Data Course Project

Spark DataFrame analytics + MLlib arrest predictor on Chicago Crime data
(Hadoop 3.4.1 / Spark 3.5.4, 1 master + 2 workers).

## Team

| Member               | ID     | GitHub             | Tasks    |
|----------------------|-------:|--------------------|----------|
| Layal Alshmassi      | 231097 | `layalalshmassi`   | 1, 5, 11 |
| Alanoud Alrowaite    | 231412 | `aalrowaite`       | 2, 6     |
| Joud Abohaimed       | 231453 | `jabohaimed`       | 3, 7, 10 |
| Aljazee Abuhemed     | 231800 | `aljazyabuhaimed`  | 4, 9     |

## Spec compliance (May 2026 update)

1. **Task 8 (CrossValidator) is omitted** — waived by the instructor.
2. **Phase B (Tasks 5–7) trains on a 5% sample** via `df.sample(fraction=0.05, seed=42)`.
   On the cluster this gives 39,534 rows (Train 31,728 / Test 7,806).
3. **Task 11 uses `--deploy-mode cluster`**. Application stdout is collected with
   `yarn logs -applicationId <appId>` into `output/spark_submit/run.log`.

## Repository layout

```
.
├── M2_Spark_ML_BigDataCourseProject.ipynb   Notebook (Tasks 1–7), executed locally
├── m2_spark_ml.py                           Standalone Phase B script for spark-submit
├── scripts/
│   └── make_notebook.py                     Notebook generator
├── output/
│   ├── year_trend.png                       Task 3 chart
│   ├── cluster_yarn_log.txt                 Task 10 evidence
│   └── spark_submit/
│       ├── console.log                      Task 11 spark-submit console output
│       └── run.log                          Task 11 application stdout
└── README.md
```

## Executive summary

We reproduce the four M1 MapReduce analyses with Spark DataFrames + Spark SQL on the
full 793,072-row HDFS dataset (numbers match M1 exactly). For arrest prediction we
build a Spark MLlib pipeline (StringIndexer × 2 + VectorAssembler + classifier) and
train Logistic Regression, Random Forest, and Gradient-Boosted Trees on a 5% sample
as required by the May 2026 spec update.

**Top model by AUC: GBT (0.8241).** Random Forest is a strong second (0.8075) and
trains 12× faster — the better trade-off for production deployment.

---

# Phase A — Spark DataFrame analytics

## Task 1 — Crime type distribution
*Layal Alshmassi (231097, `layalalshmassi`)*

```python
top_types = (crimes
             .groupBy("Primary Type")
             .agg(fns.count(fns.lit(1)).alias("freq"))
             .orderBy(fns.desc("freq")))
```

**M1 (MapReduce) ↔ M2 (Spark) — Top 10:**

| Crime type | M1 | M2 |
|------------|---:|---:|
| THEFT | 162,688 | 162,688 |
| BATTERY | 151,930 | 151,930 |
| CRIMINAL DAMAGE | 91,241 | 91,241 |
| NARCOTICS | 74,127 | 74,127 |
| ASSAULT | 54,070 | 54,070 |
| MOTOR VEHICLE THEFT | 48,494 | 48,494 |
| BURGLARY | 39,872 | 39,872 |
| OTHER OFFENSE | 36,893 | 36,893 |
| ROBBERY | 30,991 | 30,991 |
| DECEPTIVE PRACTICE | 30,396 | 30,396 |

Numbers match exactly. Spark's DataFrame engine runs the aggregation in-memory; the
streaming MapReduce equivalent had to disk-shuffle between mapper and reducer.

---

## Task 2 — Location hotspots (Spark SQL)
*Alanoud Alrowaite (231412, `aalrowaite`)*

```python
crimes.createOrReplaceTempView("chicago_crimes")

hotspots = spark.sql("""
    SELECT  `Location Description` AS spot,
            COUNT(*)               AS cases
      FROM  chicago_crimes
     WHERE  `Location Description` IS NOT NULL
     GROUP  BY `Location Description`
     ORDER  BY cases DESC
     LIMIT  10
""")
```

**M1 ↔ M2 — Top 10 hotspots (full dataset):**

| Location | M1 | M2 |
|----------|---:|---:|
| STREET | 245,437 | 248,326 |
| RESIDENCE | 136,238 | 136,393 |
| APARTMENT | 60,925 | 61,235 |
| SIDEWALK | 47,407 | 47,506 |
| OTHER | 29,213 | 29,671 |
| PARKING LOT/GARAGE(NON.RESID.) | 21,876 | 22,436 |
| ALLEY | 18,258 | 18,349 |
| SCHOOL, PUBLIC, BUILDING | 20,516 | 15,776 |
| RESIDENCE-GARAGE | 14,266 | 14,291 |
| SMALL RETAIL STORE | 13,755 | 13,804 |

Slight differences come from M1's manual CSV split dropping a few hundred edge-case
rows that Spark's CSV parser keeps.

---

# Phase B — MLlib arrest predictor (5% sample)

---

## Task 5 — Feature pipeline
*Layal Alshmassi (231097, `layalalshmassi`)*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over four
features, 80/20 split with `seed=42`. The 5% sample is applied before any feature
engineering.

Sample feature vectors from the cluster training set:

```
+-------------------+---------+--------+----+------------+-------+--------------------+-----+
|Primary Type       |pri_idx  |District|Hour|Domestic_str|dom_idx|feat_col            |label|
+-------------------+---------+--------+----+------------+-------+--------------------+-----+
|HOMICIDE           |11.0     |25      |10  |false       |0.0    |[25.0,11.0,10.0,0.0]|1    |
|HOMICIDE           |11.0     |5       |13  |false       |0.0    |[5.0,11.0,13.0,0.0] |1    |
|HOMICIDE           |11.0     |3       |20  |false       |0.0    |[3.0,11.0,20.0,0.0] |0    |
+-------------------+---------+--------+----+------------+-------+--------------------+-----+
```

Vector layout: `[District, pri_idx, Hour, dom_idx]`.

---

## Task 6 — Train and evaluate three classifiers
*Alanoud Alrowaite (231412, `aalrowaite`)*

Cluster results (5% sample of the full HDFS dataset):

| Model | Params | Train (s) | AUC | Accuracy | F1 | Precision | Recall |
|-------|--------|----------:|----:|---------:|---:|----------:|-------:|
| Logistic Regression | maxIter=100, regParam=0.01 | 24.7 | 0.6022 | 0.7280 | 0.6376 | 0.6923 | 0.7280 |
| Random Forest | numTrees=100, maxDepth=5, maxBins=64 | 39.9 | 0.8075 | 0.8156 | 0.7802 | 0.8528 | 0.8156 |
| **GBT** | maxIter=50, maxDepth=5, maxBins=64 | 463.2 | **0.8241** | **0.8500** | **0.8337** | **0.8610** | **0.8500** |

**Confusion matrices (TN/FP/FN/TP):**
- LR:  (5549, 93, 2030, 133)
- RF:  (5641, 1, 1438, 725)
- GBT: (5553, 89, 1082, 1081)

**Top model by AUC: GBT (0.8241).**

---

# Phase C — Deployment evidence

---

## Task 11 — spark-submit (cluster mode)
*Layal Alshmassi (231097, `layalalshmassi`)*

Per the May 2026 spec update, Task 11 uses `--deploy-mode cluster`:

```bash
lalshmassi@master-node:~$ spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g m2_spark_ml.py
```

YARN application: `application_1777830883738_0027` — `final status: SUCCEEDED`.

Application stdout is collected with `yarn logs -applicationId application_1777830883738_0027`
and saved to `output/spark_submit/run.log`. The console.log
(`output/spark_submit/console.log`) captures the spark-submit invocation and YARN's
progress reports.

Excerpt from `run.log`:

```
Spark version: 3.5.4
Master:        yarn
Total records: 793,072
Phase B sample: 39,534 rows  (5%, seed=42)
Train rows: 31,728 | Test rows: 7,806

>> LogisticRegression
  AUC       0.6022
  Accuracy  0.7280
  F1        0.6376
  Train(s)  40.2

>> RandomForest
  AUC       0.8075
  Accuracy  0.8156
  F1        0.7802
  Train(s)  53.0

>> GBT
  AUC       0.8241
  Accuracy  0.8500
  F1        0.8337
  Train(s)  471.1

Top model by AUC: GBT (0.8241)
```

---

## Spec note — executor cores

The M2 spec lists `--executor-cores 2`. The course YARN cluster's maximum container
allocation is `<memory:1536, vCores:1>` — requesting 2 vcores returns
`InvalidResourceRequestException`. We therefore use `--executor-cores 1`, the same
setting M1 used.

---

## Member contributions

| Member | Tasks | Contribution |
|--------|-------|--------------|
| Layal Alshmassi (`layalalshmassi`) | 1, 5, 11 | Crime-type DataFrame query; feature pipeline; spark-submit cluster-mode submission and log retrieval |
| Alanoud Alrowaite (`aalrowaite`)   | 2, 6     | Spark SQL location-hotspots query; three-classifier training and evaluation |
| Joud Abohaimed (`jabohaimed`)      | 3, 7, 10 | Year-trend table + matplotlib chart; Random Forest feature importances; yarn-client cluster execution evidence |
| Aljazee Abuhemed (`aljazyabuhaimed`) | 4, 9   | Arrest-rate analysis; local notebook execution evidence |

## How to reproduce

Locally:
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_Spark_ML_BigDataCourseProject.ipynb \
    --output M2_Spark_ML_BigDataCourseProject.ipynb
```

On the cluster:
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
# one-time deps for python3.12
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py --user
python3.12 -m pip install --user numpy 'setuptools>=68'
# Phase B standalone (cluster mode):
spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g m2_spark_ml.py
```
