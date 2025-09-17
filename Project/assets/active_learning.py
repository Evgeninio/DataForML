from dagster import op, Out
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def _uncertainty_scores(proba: np.ndarray) -> np.ndarray:
    p1 = proba[:, 1]
    return np.abs(p1 - 0.5)

@op(out=Out(object))
def active_learning_op(context, df: pd.DataFrame):
    context.log.info("Active Learning без modAL: uncertainty sampling (scikit-learn only)")

    X_text = df["text"].astype(str).values
    y = df["target"].values

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    X_all = vectorizer.fit_transform(X_text)

    rng = np.random.default_rng(42)
    seed_size = min(200, len(df))
    seed_idx = rng.choice(len(df), size=seed_size, replace=False)

    X_seed = X_all[seed_idx]
    y_seed = y[seed_idx]

    clf = LogisticRegression(max_iter=300)
    clf.fit(X_seed, y_seed)

    pool_idx = np.setdiff1d(np.arange(len(df)), seed_idx)
    iters = 5
    batch = 50
    for i in range(iters):
        if len(pool_idx) == 0:
            break

        X_pool = X_all[pool_idx]
        proba = clf.predict_proba(X_pool)

        scores = _uncertainty_scores(proba)

        take = min(batch, len(pool_idx))
        query_rel_idx = np.argsort(scores)[:take]
        query_abs_idx = pool_idx[query_rel_idx]

        X_query = X_all[query_abs_idx]
        y_query = y[query_abs_idx]

        X_aug = np.vstack([X_seed.toarray(), X_query.toarray()]) if hasattr(X_seed, "toarray") else np.vstack([X_seed, X_query])
        y_aug = np.concatenate([y_seed, y_query])

        clf.fit(X_aug, y_aug)

        X_seed = X_all[np.concatenate([seed_idx, query_abs_idx])]
        y_seed = y[np.concatenate([seed_idx, query_abs_idx])]
        seed_idx = np.concatenate([seed_idx, query_abs_idx])
        pool_idx = np.setdiff1d(pool_idx, query_abs_idx)

        context.log.info(f"AL iter {i+1}: добавили {take} примеров; осталось в пуле {len(pool_idx)}")

    final_pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    joblib.dump(final_pipe, "artifacts/active_learner_pipeline.joblib")
    context.log.info("Сохранено: artifacts/active_learner_pipeline.joblib")
    return final_pipe

