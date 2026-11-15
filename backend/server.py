"""
FastAPI bridge between the web UI and the manual NumPy Gaussian generative classifier.

Run from the project root:
    python -m uvicorn backend.server:app --port 8000
"""
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data_loader import load_and_prepare_data, get_full_dataset  # noqa: E402
from src.generative_classifier import GaussianGenerativeClassifier  # noqa: E402
from src.evaluation import evaluate_model  # noqa: E402

WEB_DIR = os.path.join(ROOT, "web")

# ---------------------------------------------------------------- model boot
X_train, X_test, y_train, y_test, FEATURES, TARGETS = load_and_prepare_data(save_csv=False)
X_all, y_all, _, _ = get_full_dataset()

_t0 = time.perf_counter()
model = GaussianGenerativeClassifier().fit(X_train, y_train)
FIT_MS = (time.perf_counter() - _t0) * 1000

# Same split parameters as load_and_prepare_data, applied to row indices.
_, _te = train_test_split(np.arange(len(X_all)), test_size=0.3, random_state=42, stratify=y_all)
TEST_IDX = {int(i) for i in _te}

CLASSES = [str(t) for t in TARGETS]
SHORT = [f.replace(" (cm)", "") for f in FEATURES]
LOG_2PI_HALF_D = 0.5 * model.d_ * np.log(2 * np.pi)

app = FastAPI(title="Iris Generative Classifier API", version="1.0")


# --------------------------------------------------------------------- utils
def _analyse(x: np.ndarray) -> dict:
    """Run every stage of the generative pipeline for a single sample."""
    x2 = x.reshape(1, -1)
    log_lik, maha, log_det = [], [], []
    for c in model.classes_:
        mu, S = model.means_[c], model.covariances_[c]
        log_lik.append(float(model._log_gaussian_density(x2, mu, S)[0]))
        d = x - mu
        maha.append(float(d @ np.linalg.solve(S, d)))
        log_det.append(float(np.linalg.slogdet(S)[1]))
    priors = [float(model.priors_[c]) for c in model.classes_]
    post = model.predict_proba(x2)[0]
    log_post = model.predict_log_proba(x2)[0]
    scores = model.decision_function(x2)[0]
    k = int(np.argmax(post))
    return {
        "input": x.tolist(),
        "prediction": CLASSES[k],
        "prediction_index": k,
        "classes": [
            {
                "name": CLASSES[i],
                "prior": priors[i],
                "log_likelihood": log_lik[i],
                "likelihood": float(np.exp(log_lik[i])),
                "mahalanobis_sq": maha[i],
                "log_det": log_det[i],
                "joint": float(np.exp(log_lik[i]) * priors[i]),
                "posterior": float(post[i]),
                "log_posterior": float(log_post[i]),
                "discriminant": float(scores[i]),
            }
            for i in range(len(CLASSES))
        ],
        "margin": float(np.sort(post)[-1] - np.sort(post)[-2]),
    }


class PredictIn(BaseModel):
    sepal_length: float = Field(..., ge=0, le=15)
    sepal_width: float = Field(..., ge=0, le=15)
    petal_length: float = Field(..., ge=0, le=15)
    petal_width: float = Field(..., ge=0, le=15)

    def vec(self):
        return np.array([self.sepal_length, self.sepal_width, self.petal_length, self.petal_width], dtype=float)


# ----------------------------------------------------------------------- API
@app.get("/api/meta")
def meta():
    lo, hi = X_all.min(axis=0), X_all.max(axis=0)
    return {
        "features": SHORT,
        "classes": CLASSES,
        "fit_ms": FIT_MS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X_all)),
        "ranges": [{"min": float(a), "max": float(b)} for a, b in zip(lo, hi)],
        "model": [
            {
                "name": CLASSES[i],
                "prior": float(model.priors_[c]),
                "mean": model.means_[c].tolist(),
                "covariance": model.covariances_[c].tolist(),
            }
            for i, c in enumerate(model.classes_)
        ],
    }


@app.post("/api/predict")
def predict(body: PredictIn):
    return _analyse(body.vec())


@app.get("/api/dataset")
def dataset():
    preds = model.predict(X_all)
    return {
        "samples": [
            {"x": X_all[i].tolist(), "label": int(y_all[i]), "pred": int(preds[i]), "test": i in TEST_IDX}
            for i in range(len(X_all))
        ]
    }


@app.get("/api/evaluation")
def evaluation():
    res = evaluate_model(model, X_test, y_test, TARGETS)
    rep = classification_report(y_test, res["y_pred"], target_names=TARGETS, output_dict=True)
    train_acc = float(np.mean(model.predict(X_train) == y_train))
    return {
        "accuracy": float(res["accuracy"]),
        "train_accuracy": train_acc,
        "confusion_matrix": res["confusion_matrix"].tolist(),
        "report": {k: v for k, v in rep.items() if k in CLASSES},
        "test": [
            {
                "x": X_test[i].tolist(),
                "label": int(y_test[i]),
                "pred": int(res["y_pred"][i]),
                "confidence": float(model.predict_proba(X_test[i : i + 1]).max()),
            }
            for i in range(len(X_test))
        ],
    }


@app.get("/api/field")
def field(
    fx: int = Query(2, ge=0, le=3),
    fy: int = Query(3, ge=0, le=3),
    nx: int = Query(72, ge=8, le=200),
    ny: int = Query(48, ge=8, le=200),
):
    """
    Posterior map over a 2-D feature slice. A Gaussian's marginal over a subset of
    dimensions is the Gaussian of the corresponding mean/covariance sub-block, so the
    map is exact for the question "how would the model vote knowing only these two traits".
    """
    if fx == fy:
        raise HTTPException(400, "fx and fy must differ")
    lo, hi = X_all.min(axis=0), X_all.max(axis=0)
    pad = 0.12
    x0, x1 = lo[fx] - pad * (hi[fx] - lo[fx]), hi[fx] + pad * (hi[fx] - lo[fx])
    y0, y1 = lo[fy] - pad * (hi[fy] - lo[fy]), hi[fy] + pad * (hi[fy] - lo[fy])
    gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    P = np.stack([gx.ravel(), gy.ravel()], axis=1)
    idx = [fx, fy]
    logs = []
    for c in model.classes_:
        mu = model.means_[c][idx]
        S = model.covariances_[c][np.ix_(idx, idx)]
        _, logdet = np.linalg.slogdet(S)
        D = P - mu
        m = np.sum(D * np.linalg.solve(S, D.T).T, axis=1)
        logs.append(-np.log(2 * np.pi) - 0.5 * logdet - 0.5 * m + np.log(model.priors_[c]))
    L = np.stack(logs, axis=1)
    L -= L.max(axis=1, keepdims=True)
    post = np.exp(L)
    post /= post.sum(axis=1, keepdims=True)
    return {
        "bounds": [float(x0), float(x1), float(y0), float(y1)],
        "nx": nx,
        "ny": ny,
        "posterior": np.round(post, 4).reshape(ny, nx, -1).tolist(),
    }


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-cache"  # always revalidate; files are tiny
    return resp


# -------------------------------------------------------------- static front
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
