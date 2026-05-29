from pyspark.ml.evaluation import RegressionEvaluator
from q2_share import OUTPUT_DIR, build_pipeline, create_spark, load_or_prepare_model_df, split_train_test

# SCRIPT TO DEFINE THE SPARK ML PIPELINE AND DO A SINGLE FIT

def main():
    spark = create_spark("macs30123_a3_q2b")
    df = load_or_prepare_model_df(spark)
    train_df, test_df = split_train_test(df)

    pipeline = build_pipeline(reg_param=0.01, elastic_net_param=0.0)
    pipeline_model = pipeline.fit(train_df)
    predictions = pipeline_model.transform(test_df)

    evaluator = RegressionEvaluator(
        labelCol="tip_amount",
        predictionCol="prediction",
        metricName="rmse",
    )
    rmse = evaluator.evaluate(predictions)

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "q2b_summary.txt"
    with summary_path.open("w") as f:
        f.write("Assignment 3 Q2b summary\n")
        f.write("Pipeline stages:\n")
        for idx, stage in enumerate(pipeline.getStages()):
            f.write(f"- stage_{idx}: {stage.__class__.__name__}\n")
        f.write("\nThis pipeline can be passed directly into a Spark CrossValidator.\n")
        f.write(f"Single-fit RMSE with fixed params (regParam=0.01, elasticNetParam=0.0): {rmse}\n")

    print("Pipeline stages:")
    for idx, stage in enumerate(pipeline.getStages()):
        print(f"  stage_{idx}: {stage.__class__.__name__}")
    print(f"\nSingle-fit RMSE: {rmse}")
    print(f"Saved q2b summary to: {summary_path.resolve()}")

    spark.stop()

if __name__ == "__main__":
    main()
