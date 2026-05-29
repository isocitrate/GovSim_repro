import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

DATA_PATH = "/project/macs30123/nyc-tlc/*.parquet"
N_WORKERS = 1
THREADS_PER_WORKER = 1
MEMORY_LIMIT = "16GB"
HEAD_ROWS = 5
SAMPLE_FRAC = 0.1 #(Set to float \in [0, 1] to sample fraction, or None for full)

def start_client():
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=MEMORY_LIMIT,
    )
    return Client(cluster)

def load_data(path = DATA_PATH, sample_frac = SAMPLE_FRAC):
    df = dd.read_parquet(path, engine = "pyarrow", split_row_groups = True)
    if sample_frac is not None:
        df = df.sample(frac = sample_frac, random_state = 12345)
    return df

def show_overview(df, head_rows=HEAD_ROWS):
    print(f"Path: {DATA_PATH}")
    print(f"Partitions: {df.npartitions}")
    print(f"Columns ({len(df.columns)}):")
    for column in df.columns:
        print(f"  - {column}")

    print("\nDtypes:")
    print(df.dtypes.to_string())

    print(f"\nPreview ({head_rows} rows):")
    print(df.head(head_rows).to_string())

if __name__ == "__main__":
    client = start_client()
    print(client)

    df = load_data()
    show_overview(df)