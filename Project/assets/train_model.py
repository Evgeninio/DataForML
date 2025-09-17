
from dagster import op, Out
import pandas as pd
import joblib, os
from sklearn.metrics import classification_report
import numpy as np

@op(out=Out(str))
def train_model_op(context, model, df: pd.DataFrame = None):
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")

    report_path = "artifacts/entropy_report.csv"
    if df is not None and "entropy" in df.columns:
        df[["text", "entropy", "prob_spam", "target"]].to_csv(report_path, index=False)
        context.log.info(f"Saved entropy report to {report_path}")

    context.log.info("Saved final model to artifacts/model.joblib")
    return "artifacts/model.joblib"
