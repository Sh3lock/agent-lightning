import pandas as pd
import os
from langchain_community.utilities import SQLDatabase
from tqdm import tqdm

# Path setup
BASE_DIR = '/root/verl_pass/agent-lightning/examples/spider'
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_DIR = os.path.join(DATA_DIR, 'database')
PARQUET_FILE = os.path.join(DATA_DIR, 'train_spider.parquet')

# Prompt template (simplified for length estimation)
SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
     Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```
"""

USER_PROMPT = "Question: {input}"

def get_prompt_length(row):
    db_id = row['db_id']
    question = row['question']
    
    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")
    uri = f"sqlite:///{db_path}"
    
    try:
        db = SQLDatabase.from_uri(uri)
        table_info = db.get_table_info()
        
        # Construct full prompt string
        system_part = SYSTEM_PROMPT.format(dialect="sqlite", table_info=table_info)
        user_part = USER_PROMPT.format(input=question)
        
        full_text = system_part + "\n" + user_part
        return len(full_text)
    except Exception as e:
        print(f"Error processing {db_id}: {e}")
        return 0

def main():
    print("Loading data...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df)} rows.")
    
    lengths = []
    print("Calculating lengths...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        length = get_prompt_length(row)
        lengths.append(length)
    
    df['prompt_length'] = lengths
    
    # Sort by length descending
    df_sorted = df.sort_values('prompt_length', ascending=False)
    
    # Take top 48
    top_48 = df_sorted.head(48)
    
    print("Top 5 lengths (chars):")
    print(top_48['prompt_length'].head().values)
    
    output_file = os.path.join(DATA_DIR, 'reproduce_oom.parquet')
    top_48.to_parquet(output_file)
    print(f"Saved top 48 longest examples to {output_file}")

if __name__ == "__main__":
    main()
