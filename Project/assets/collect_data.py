
from dagster import op, Out
import os, io, zipfile, requests, pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

@op(out=Out(pd.DataFrame))
def collect_sms_data_op(context):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    context.log.info(f"Downloading dataset from {UCI_ZIP_URL}")
    r = requests.get(UCI_ZIP_URL, timeout=60)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open("SMSSpamCollection") as f:
        df = pd.read_csv(f, sep="\t", header=None, names=["label", "text"])

    df["target"] = (df["label"] == "spam").astype(int)

    raw_path = "data/raw/sms_spam.csv"
    df.to_csv(raw_path, index=False)
    context.log.info(f"Saved raw dataset to {raw_path} (rows={len(df)})")
    return df
