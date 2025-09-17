
from dagster import op, Out
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
import numpy as np

@op(out=Out(pd.DataFrame))
def preprocess_and_entropy_op(context, df: pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=20000)),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    pipe.fit(train_df["text"], train_df["target"])
    probs = pipe.predict_proba(df["text"])

    eps = 1e-12
    ent = -(probs * np.log2(probs + eps)).sum(axis=1)

    out_df = df.copy()
    out_df["entropy"] = ent
    out_df["prob_spam"] = probs[:, 1]

    ll = log_loss(test_df["target"], pipe.predict_proba(test_df["text"]))
    context.log.info(f"Initial model log_loss on holdout: {ll:.4f}")

    import joblib, os
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipe, "artifacts/base_pipeline.joblib")

    proc_path = "data/processed/sms_spam_with_entropy.parquet"
    out_df.to_parquet(proc_path, index=False)
    context.log.info(f"Saved processed data with entropy to {proc_path} (rows={len(out_df)})")
    return out_df
