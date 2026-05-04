"""SE446 Milestone 2 — Big Data Course Project

Phase B (Tasks 5-7): standalone Spark MLlib pipeline for the spark-submit deliverable.

Authors:
    Task 5 — Layal Alshmassi (231097)
    Task 6 — Alanoud Alrowaite (231412)
    Task 7 — Joud Abohaimed (231453)

Per the May 2026 spec update:
    * Task 8 (CrossValidator) is waived.
    * Phase B trains on a 5% sample (df.sample(fraction=0.05, seed=42)).

Submit via:
    spark-submit \\
        --master yarn --deploy-mode cluster \\
        --num-executors 2 --executor-memory 1g --executor-cores 1 \\
        m2_spark_ml.py
"""
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as fns
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)


HDFS_FILE = "hdfs:///data/chicago_crimes.csv"


def open_session() -> SparkSession:
    return (SparkSession.builder
            .appName("M2_BigDataCourseProject_m2_spark_ml")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def read_crimes(session: SparkSession):
    raw = session.read.csv(HDFS_FILE, header=True, inferSchema=True)
    enriched = (raw
                .withColumn("Hour",
                            fns.hour(fns.to_timestamp(fns.col("Date"),
                                                      "MM/dd/yyyy hh:mm:ss a")))
                .withColumn("label",        fns.col("Arrest").cast(IntegerType()))
                .withColumn("Domestic_str", fns.col("Domestic").cast(StringType())))
    return enriched.dropna(subset=["District", "Primary Type",
                                   "Hour", "Domestic_str", "label"])


def evaluate_predictions(predictions, bin_eval, multi_eval):
    return {
        "AUC":       bin_eval.evaluate(predictions),
        "Accuracy":  multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"}),
        "F1":        multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}),
        "Precision": multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}),
        "Recall":    multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}),
    }


def confusion_box(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    box = {(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}
    return (box.get((0, 0), 0), box.get((0, 1), 0),
            box.get((1, 0), 0), box.get((1, 1), 0))


def main():
    spark = open_session()
    print("Spark version:", spark.version)
    print("Master:       ", spark.sparkContext.master)

    crimes = read_crimes(spark)
    print("Total records:", f"{crimes.count():,}")

    # ----- Task 5 (Layal): pipeline + 5% sample -----
    sub = crimes.sample(fraction=0.05, seed=42)
    print("Phase B sample:", f"{sub.count():,} rows  (5%, seed=42)")

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

    train_data, test_data = sub.randomSplit([0.8, 0.2], seed=42)
    train_data.cache()
    test_data.cache()
    print("Train rows:", f"{train_data.count():,}", "| Test rows:", f"{test_data.count():,}")

    bin_eval   = BinaryClassificationEvaluator(labelCol="label")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label",
                                                   predictionCol="prediction")

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

    fitted_rf = None
    bag = []
    for tag, learner in classifiers:
        pipeline = Pipeline(stages=[pri_idx, dom_idx, feat_col, learner])
        t0 = time.time()
        fitted = pipeline.fit(train_data)
        elapsed = time.time() - t0
        preds = fitted.transform(test_data)
        metrics = evaluate_predictions(preds, bin_eval, multi_eval)
        cm = confusion_box(preds)
        bag.append((tag, elapsed, metrics, cm))
        print(f"\n>> {tag}")
        for k, v in metrics.items():
            print(f"  {k:<10}{v:.4f}")
        print(f"  Train(s)  {elapsed:.1f}")
        print(f"  CM (TN,FP,FN,TP) = {cm}")
        if tag == "RandomForest":
            fitted_rf = fitted.stages[-1]

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'metric':<11}{'Logistic':>14}{'RandomForest':>16}{'GBT':>14}")
    print("-" * 78)
    for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
        print(f"{k:<11}{bag[0][2][k]:>14.4f}{bag[1][2][k]:>16.4f}{bag[2][2][k]:>14.4f}")
    print(f"{'Train(s)':<11}{bag[0][1]:>14.1f}{bag[1][1]:>16.1f}{bag[2][1]:>14.1f}")
    print("=" * 78)
    top = max(bag, key=lambda r: r[2]["AUC"])
    print("Top model by AUC:", top[0], f"({top[2]['AUC']:.4f})")

    # ----- Task 7 (Joud): RF feature importances -----
    print("\n--- Random Forest feature importances ---")
    layout = ["District", "pri_idx", "Hour", "dom_idx"]
    for feat, imp in sorted(zip(layout, fitted_rf.featureImportances.toArray()),
                            key=lambda kv: -kv[1]):
        print(f"  {feat:<10} {imp:.4f}  {'=' * int(round(imp * 50))}")

    spark.stop()


if __name__ == "__main__":
    main()
