"""Optional reconstruction from QMSum and the pinned encoder (or its cached embeddings)."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from . import endpoint_features as features
from controller.index import indexed

from .paths import AUDIT, DESIGN, OUTPUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assets",
        type=Path,
        help="Directory containing qmsum.zip, minilm/ and optionally embeddings/",
    )
    args = parser.parse_args()

    cache = args.assets / "embeddings"

    def encode(texts):
        nonlocal encoder
        if encoder is None:
            encoder = features.Encoder(args.assets)
        return encoder(texts)

    encoder = None
    with zipfile.ZipFile(args.assets / "qmsum.zip") as f:
        docs = [
            (Path(n).stem, json.loads(f.read(n)))
            for split in ("train", "val", "test")
            for n in sorted(f.namelist())
            if "/Product/" + split + "/" in n and n.endswith(".json")
        ]
    index = json.loads((DESIGN / "query_index.json").read_text())
    roles = json.loads((DESIGN / "calibration.json").read_text())
    with np.load(AUDIT / "endpoints.npz") as z:
        offset = z["offset"]
        values = z["values"]
        assert len(index) == sum(z["domain"] == "design")
    report = []
    matrices = []
    for fold, role in enumerate(roles):
        fit = [m for m in docs if m[0][:-1] in role["endpoint_fit"]]
        test = [m for m in docs if m[0][:-1] in role["test"]]
        vectorizers = features.vecfit(fit)
        tr = features.build(fit, vectorizers)
        te = features.build(test, vectorizers)
        features.enrich(tr, fit, encode, args.assets)
        features.enrich(te, test, encode, args.assets)
        wt = np.concatenate([np.full(len(q["y"]), 1 / len(q["y"])) for q in tr])
        y = np.concatenate([q["y"] for q in tr])
        rec = dict(
            fold=fold,
            training_ids=[q["id"] for q in tr],
            parameters={},
            max_error={},
            graphs_equal=0,
        )
        for feat, key, col in [("l", "b", 1), ("e", "r", 0)]:
            X = np.vstack([q[feat] for q in tr])
            sc = StandardScaler().fit(X)
            mdl = LogisticRegression(C=1, max_iter=600).fit(
                sc.transform(X), y, sample_weight=wt / wt.mean()
            )
            rec["parameters"][key] = dict(
                mean=sc.mean_.tolist(),
                scale=sc.scale_.tolist(),
                coef=mdl.coef_[0].tolist(),
                intercept=float(mdl.intercept_[0]),
            )
            for q in te:
                q[key] = mdl.predict_proba(sc.transform(q[feat]))[:, 1]
        byid = {q["id"]: q for q in te}
        for i, record in enumerate(index):
            if record["fold"] != fold:
                continue
            q = byid[record["id"]]
            r, b, p, y = values[offset[i] : offset[i + 1]].T
            np.testing.assert_array_equal(q["y"], y)
            for key, target in [("r", r), ("b", b)]:
                err = float(np.max(abs(q[key] - target)))
                rec["max_error"][key] = max(err, rec["max_error"].get(key, 0))
            assert indexed(q["r"], q["b"])[1] == indexed(r, b)[1]
            rec["graphs_equal"] += 1
            matrices.append((i, np.column_stack([q["l"], q["e"][:, 10:]])))
        assert max(rec["max_error"].values()) < 1e-9
        report.append(rec)
        print(fold, rec["graphs_equal"], rec["max_error"], flush=True)
    (OUTPUT / "endpoint_fits.json").write_text(
        json.dumps(
            dict(
                sklearn=sklearn.__version__,
                encoder_revision=(args.assets / "minilm/revision.txt").read_text().strip(),
                folds=report,
            ),
            indent=1,
        )
    )
    np.savez_compressed(
        OUTPUT / "heldout_features.npz",
        X=np.vstack([x for _, x in sorted(matrices, key=lambda v: v[0])]),
    )


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
