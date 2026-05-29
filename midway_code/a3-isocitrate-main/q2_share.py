from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast
from pyspark.ml import Pipeline
from pyspark.ml.feature import Bucketizer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression

# SHARED HELPER CODE FOR Q2 and Q3

TRIP_DATA_PATH = "/project/macs30123/nyc-tlc/"
ZONE_LOOKUP_PATH = "taxi_zone_lookup.csv"
OUTPUT_DIR = Path("q2_outputs")
PREPARED_DATA_PATH = OUTPUT_DIR / "model_df.parquet"
DEBUG_FRACTION = None # Float \in [0, 1] if debug, else None"
TRAIN_FRACTION = 0.8
SEED = 12345

# Add explanatory notes for engineered features
FEATURE_NOTES = {
    "pickup_borough": "Categorical geography feature capturing broad pickup context.",
    "same_borough": "Spatial binary indicating whether pickup and dropoff occur in the same borough.",
    "airport_trip": "Spatial binary for airport-linked rides.",
    "pickup_hour": "Datetime feature capturing hourly variations in travel/traffic.",
    "is_weekend": "Datetime binary distinguishing weekdays and weekends.",
    "passenger_count": "Integer capturing party size and potentially different tipping norms.",
    "fare_amount": "Float capturing trip price (sans the tip).",
    "fare_per_mile": "Float capturing general pricing intensity.",
    "trip_duration_min": "Float proxying congestion and trip burden.",
    "trip_distance_bucket": "Categorical bucketized trip distance to capture potential nonlinear thresholds.",
}

# Function to check availability of lookup table
def ensure_zone_lookup():
    path = Path(ZONE_LOOKUP_PATH)
    if path.exists():
        print(f"Using existing zone lookup file: {path.resolve()}")
        return str(path)
    else: 
        raise RuntimeError(
            print(f"Zone lookup file not found, please download and upload to midway")
        )

# Function to start spark session
def create_spark(app_name):
    return (
        SparkSession.builder.appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", "12g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.parquet.enableVectorizedReader", "false")
        .config("spark.sql.files.maxPartitionBytes", "33554432")
        .getOrCreate()
    )

# Function to load lookup table
def load_zone_lookup(spark, zone_lookup_path):
    zones = spark.read.csv(zone_lookup_path, header=True, inferSchema=True)
    return zones.select("LocationID", "Borough", "Zone")

# Function to load trip data
def load_trips(spark):
    df = spark.read.parquet(TRIP_DATA_PATH)
    cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "tip_amount",
    ]
    df = df.select(*cols)

    if DEBUG_FRACTION is not None:
        df = df.sample(withReplacement=False, fraction=DEBUG_FRACTION, seed=SEED)

    return df

# Basic processing to subset only columns needed and remove invalids
def clean_trips(df):
    df = df.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "PULocationID",
            "DOLocationID",
            "fare_amount",
            "tip_amount",
        ]
    )

    return df.filter(
        (F.col("passenger_count") > 0)
        & (F.col("trip_distance") > 0)
        & (F.col("fare_amount") > 0)
        & (F.col("tip_amount") >= 0)
    )

# Add pickup hour, weekdays, and trip duration bins to the features
def add_time_features(df):
    duration_minutes = (
        F.unix_timestamp("tpep_dropoff_datetime")
        - F.unix_timestamp("tpep_pickup_datetime")
    ) / 60.0

    df = (
        df.withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
        .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))
        .withColumn(
            "is_weekend",
            F.when(F.col("pickup_dayofweek").isin([1, 7]), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("trip_duration_min", duration_minutes)
    )

    return df.filter((F.col("trip_duration_min") > 0) & (F.col("trip_duration_min") <= 180))

# Add boroughs, cross-borough trips, and airport trip indicators to the features
def add_geo_features(df, zones):
    pickup_zones = zones.select(
        F.col("LocationID").alias("pu_location_id"),
        F.col("Borough").alias("pickup_borough"),
        F.col("Zone").alias("pickup_zone"),
    )
    dropoff_zones = zones.select(
        F.col("LocationID").alias("do_location_id"),
        F.col("Borough").alias("dropoff_borough"),
        F.col("Zone").alias("dropoff_zone"),
    )

    df = df.join(
        broadcast(pickup_zones),
        df["PULocationID"] == pickup_zones["pu_location_id"],
        how="left",
    ).join(
        broadcast(dropoff_zones),
        df["DOLocationID"] == dropoff_zones["do_location_id"],
        how="left",
    )

    airport_condition = (
        F.lower(F.coalesce(F.col("pickup_zone"), F.lit(""))).contains("airport")
        | F.lower(F.coalesce(F.col("dropoff_zone"), F.lit(""))).contains("airport")
        | (F.col("pickup_zone") == F.lit("EWR"))
        | (F.col("dropoff_zone") == F.lit("EWR"))
    )

    return (
        df.withColumn(
            "same_borough",
            F.when(F.col("pickup_borough") == F.col("dropoff_borough"), F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn("airport_trip", F.when(airport_condition, F.lit(1)).otherwise(F.lit(0)))
    )

# Add fare/mile to the features
def add_trip_features(df):
    df = df.withColumn("fare_per_mile", F.col("fare_amount") / F.col("trip_distance"))
    return df.filter((F.col("fare_per_mile") > 0) & (F.col("fare_per_mile") <= 100))

# FInal clean before training
def finalize_model_df(df):
    keep_cols = [
        "pickup_borough",
        "same_borough",
        "airport_trip",
        "pickup_hour",
        "is_weekend",
        "passenger_count",
        "fare_amount",
        "fare_per_mile",
        "trip_duration_min",
        "trip_distance",
        "tip_amount",
    ]
    return df.select(*keep_cols).dropna()

# Main function to prep the model
def prepare_model_df(spark):
    zone_lookup_path = ensure_zone_lookup()
    zones = load_zone_lookup(spark, zone_lookup_path)
    trips = load_trips(spark)
    df = clean_trips(trips)
    df = add_time_features(df)
    df = add_geo_features(df, zones)
    df = add_trip_features(df)
    return finalize_model_df(df)

# Save the prepped model data to disk
def save_prepared_model_df(df):
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.write.mode("overwrite").parquet(str(PREPARED_DATA_PATH))

# Load the prepared model data (or create it)
def load_or_prepare_model_df(spark):
    if PREPARED_DATA_PATH.exists():
        print(f"Using prepared model data: {PREPARED_DATA_PATH.resolve()}")
        return spark.read.parquet(str(PREPARED_DATA_PATH))

    df = prepare_model_df(spark)
    save_prepared_model_df(df)
    return spark.read.parquet(str(PREPARED_DATA_PATH))


def split_train_test(df):
    return df.randomSplit([TRAIN_FRACTION, 1 - TRAIN_FRACTION], seed=SEED)

# Build spark ML pipeline
def build_pipeline(reg_param=0.01, elastic_net_param=0.0):
    borough_indexer = StringIndexer(
        inputCol="pickup_borough",
        outputCol="pickup_borough_index",
        handleInvalid="keep",
    )
    borough_encoder = OneHotEncoder(
        inputCols=["pickup_borough_index"],
        outputCols=["pickup_borough_ohe"],
        handleInvalid="keep",
    )
    distance_bucketizer = Bucketizer(
        splits=[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, float("inf")],
        inputCol="trip_distance",
        outputCol="trip_distance_bucket",
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=[
            "pickup_borough_ohe",
            "same_borough",
            "airport_trip",
            "pickup_hour",
            "is_weekend",
            "passenger_count",
            "fare_amount",
            "fare_per_mile",
            "trip_duration_min",
            "trip_distance_bucket",
        ],
        outputCol="features",
        handleInvalid="skip",
    )
    lr = LinearRegression(
        featuresCol="features",
        labelCol="tip_amount",
        predictionCol="prediction",
        regParam=reg_param,
        elasticNetParam=elastic_net_param,
    )
    return Pipeline(stages=[borough_indexer, borough_encoder, distance_bucketizer, assembler, lr])

# Access feature names later for easier interp.
def get_feature_names(best_model):
    borough_indexer_model = best_model.stages[0]
    borough_ohe_model = best_model.stages[1]
    borough_labels = list(borough_indexer_model.labels)

    borough_size = borough_ohe_model.categorySizes[0]
    if borough_ohe_model.getDropLast():
        borough_size -= 1

    borough_feature_names = [f"pickup_borough={label}" for label in borough_labels[:borough_size]]
    while len(borough_feature_names) < borough_size:
        borough_feature_names.append("pickup_borough=__unknown__")

    other_feature_names = [
        "same_borough",
        "airport_trip",
        "pickup_hour",
        "is_weekend",
        "passenger_count",
        "fare_amount",
        "fare_per_mile",
        "trip_duration_min",
        "trip_distance_bucket",
    ]

    return borough_feature_names + other_feature_names


def build_coefficient_table(best_model, lr_model):
    feature_names = get_feature_names(best_model)
    coefficients = list(lr_model.coefficients)
    rows = []
    for name, coef in zip(feature_names, coefficients):
        rows.append(
            {
                "feature": name,
                "coefficient": float(coef),
                "abs_coefficient": abs(float(coef)),
            }
        )
    return sorted(rows, key=lambda x: x["abs_coefficient"], reverse=True)
