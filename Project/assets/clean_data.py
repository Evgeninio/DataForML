
from dagster import op, Out
import pandas as pd

@op(out=Out(pd.DataFrame))
def clean_data_op(context, df: pd.DataFrame):
    before = len(df)
    df = df.drop_duplicates(subset=["text"])

    df = df[df["text"].str.len().between(3, 300)]

    after = len(df)
    context.log.info(f"Cleaned data: {before} -> {after}")
    df.to_parquet("data/processed/clean_sms_spam.parquet", index=False)
    return df
