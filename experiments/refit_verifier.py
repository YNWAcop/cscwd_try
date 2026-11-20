"""Refit the five source-team probability mappings from saved MiniLM logits."""
import json
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .paths import DESIGN, AUDIT, OUTPUT


def main():
    queries = json.loads((DESIGN / "query_index.json").read_text())
    folds = json.loads((DESIGN / "calibration.json").read_text())
    with np.load(DESIGN / "verifier_logits.npz") as archive:
        logits = archive["logits"]
    with np.load(AUDIT / "endpoints.npz") as archive:
        offsets = archive["offset"][: len(queries) + 1]
        values = archive["values"][: offsets[-1]]
    report = []
    for fold in folds:
        fit = np.zeros(len(logits), dtype=bool)
        test = np.zeros(len(logits), dtype=bool)
        weights = np.zeros(len(logits))
        for q, start, end in zip(queries, offsets[:-1], offsets[1:]):
            team = q["meeting"][:-1]
            fit[start:end] = team in fold["verifier_fit"]
            test[start:end] = team in fold["test"]
            weights[start:end] = 1 / (end - start)
        scaler = StandardScaler().fit(logits[fit, None])
        weights_fit = weights[fit]
        model = LogisticRegression(C=1, max_iter=600).fit(
            scaler.transform(logits[fit, None]), values[fit, 3],
            sample_weight=weights_fit / weights_fit.mean(),
        )
        slope = model.coef_[0, 0] / scaler.scale_[0]
        bias = model.intercept_[0] - slope * scaler.mean_[0]
        predicted = expit(slope * logits[test].astype(float) + bias)
        error = float(np.max(np.abs(predicted - values[test, 2])))
        assert error < 2e-5
        report.append(dict(fold=fold["fold"], slope=slope, intercept=bias, max_error=error))
    (OUTPUT / "verifier_calibration.json").write_text(json.dumps(report, indent=2))
    print(f"Verified {len(report)} probability mappings.")


if __name__ == "__main__":
    main()
