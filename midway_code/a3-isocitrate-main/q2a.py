from q2_share import OUTPUT_DIR, FEATURE_NOTES, create_spark, prepare_model_df, save_prepared_model_df

# SCRIPT TO CLEAN THE MODEL DATA AND ADD NEEDED FEATURES

def main():
    spark = create_spark("macs30123_a3_q2a")
    df = prepare_model_df(spark)

    OUTPUT_DIR.mkdir(exist_ok=True)
    save_prepared_model_df(df)

    row_count = df.count()
    summary_path = OUTPUT_DIR / "q2a_summary.txt"
    with summary_path.open("w") as f:
        f.write("Assignment 3 Q2a summary\n")
        f.write("Engineered features:\n")
        for name, note in FEATURE_NOTES.items():
            f.write(f"- {name}: {note}\n")
        f.write("\nFinal modeling schema:\n")
        f.write(df._jdf.schema().treeString())
        f.write(f"\nRow count after cleaning: {row_count}\n")

    print("\nFinal modeling schema:")
    df.printSchema()
    print(f"\nRow count after cleaning: {row_count}")
    print(f"Saved prepared dataset to: {OUTPUT_DIR.resolve()}")
    print(f"Saved q2a summary to: {summary_path.resolve()}")

    spark.stop()

if __name__ == "__main__":
    main()
