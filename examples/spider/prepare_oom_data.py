import pandas as pd
from langchain_community.utilities import SQLDatabase
import os

# Path to data
DATA_DIR = "/home/storage/wenbinxing/ltf/passk/agent-lightning/examples/spider/data"
PARQUET_FILE = os.path.join(DATA_DIR, "train_spider.parquet")
DB_DIR = os.path.join(DATA_DIR, "database")

# Read parquet
print(f"Reading {PARQUET_FILE}...")
df = pd.read_parquet(PARQUET_FILE)

# Function to get schema length
def get_schema_length(db_id):
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    uri = f"sqlite:///{db_path}"
    try:
        # We use include_tables=None to get all tables, which is the default behavior usually
        db = SQLDatabase.from_uri(uri)
        table_info = db.get_table_info()
        return len(table_info)
    except Exception as e:
        print(f"Error reading {db_id}: {e}")
        return 0

# Add length column
schema_lengths = {}
unique_dbs = df['db_id'].unique()
print(f"Found {len(unique_dbs)} unique databases. Calculating schema lengths...")

for i, db_id in enumerate(unique_dbs):
    if i % 10 == 0:
        print(f"Processing {i}/{len(unique_dbs)}...", end='\r')
    schema_lengths[db_id] = get_schema_length(db_id)
print("\nDone calculating schema lengths.")

df['schema_len'] = df['db_id'].map(schema_lengths)
df['question_len'] = df['question'].str.len()
# Approximate total length (characters). 
# Note: Token count is roughly char count / 3 or 4, but relative order should be similar.
df['total_len'] = df['schema_len'] + df['question_len']

# Sort by total_len descending
df_sorted = df.sort_values('total_len', ascending=False)

# Take top 48
df_top48 = df_sorted.head(48)

# Save
output_path = os.path.join(DATA_DIR, "reproduce_oom.parquet")
df_top48.to_parquet(output_path)
print(f"Saved top 48 longest sequences to {output_path}")
print("Top 5 longest:")
print(df_top48[['db_id', 'question', 'total_len']].head())
