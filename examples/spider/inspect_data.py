import pandas as pd

try:
    df = pd.read_parquet('/root/verl_pass/agent-lightning/examples/spider/data/train_spider.parquet')
    print(df.columns)
    print(df.iloc[0])
except Exception as e:
    print(e)
