import json
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from q2_share import OUTPUT_DIR, SEED, build_coefficient_table, build_pipeline, create_spark, load_or_prepare_model_df, split_train_test

# Script for hyperparameter tuning

def main():
    spark = create_spark("macs30123_a3_q3")
    df = load_or_prepare_model_df(spark)
    train_df, test_df = split_train_test(df)

    pipeline = build_pipeline()
    lr = pipeline.getStages()[-1]

    evaluator = RegressionEvaluator(
        labelCol="tip_amount",
        predictionCol="prediction",
        metricName="rmse",
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [round(x * 0.01, 2) for x in range(10)])
        .addGrid(lr.elasticNetParam, [0.0, 1.0])
        .build()
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,
        seed=SEED,
        parallelism=1,
    )

    cv_model = cv.fit(train_df)
    predictions = cv_model.transform(test_df)
    rmse = evaluator.evaluate(predictions)

    best_model = cv_model.bestModel
    lr_model = best_model.stages[-1]
    coefficient_table = build_coefficient_table(best_model, lr_model)

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "q3_summary.txt"
    json_path = OUTPUT_DIR / "q3_coefficients.json"

    with summary_path.open("w") as f:
        f.write("Assignment 3 Q3 summary\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"Best regParam: {lr_model.getRegParam()}\n")
        f.write(f"Best elasticNetParam: {lr_model.getElasticNetParam()}\n")
        f.write(f"Intercept: {lr_model.intercept}\n")
        f.write("Coefficients (sorted by absolute value):\n")
        for row in coefficient_table:
            f.write(f"  {row['feature']}: {row['coefficient']}\n")

    with json_path.open("w") as f:
        json.dump(coefficient_table, f, indent=2)

    print("\nQ3 results:")
    print(f"RMSE: {rmse}")
    print(f"Best regParam: {lr_model.getRegParam()}")
    print(f"Best elasticNetParam: {lr_model.getElasticNetParam()}")
    print("\nTop features by absolute coefficient:")
    for row in coefficient_table[:10]:
        print(f"  {row['feature']}: {row['coefficient']}")
    print(f"\nSaved q3 summary to: {summary_path.resolve()}")
    print(f"Saved q3 coefficients to: {json_path.resolve()}")

    spark.stop()


if __name__ == "__main__":
    main()
