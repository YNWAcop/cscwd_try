"""Cold re-encoding of the manuscript example, checked against its original embeddings."""
from pathlib import Path
import sys
import json
import zipfile
import hashlib
import numpy as np
from . import endpoint_features as ef
from .paths import DESIGN, OUTPUT

root = Path(sys.argv[1])
meeting = "ES2003c"
with zipfile.ZipFile(root / "qmsum.zip") as f:
    doc = json.loads(
        f.read(
            next(
                n for n in f.namelist() if "/Product/" in n and n.endswith("/" + meeting + ".json")
            )
        )
    )
turns = [t["speaker"] + ": " + t["content"] for t in doc["meeting_transcripts"]]
chunks = [" ".join(turns[k : k + 8]) for k in range(0, len(turns), 8)]
pairs = [" ".join(turns[k : k + 2]) for k in range(0, len(turns), 2)]
roles = json.loads((DESIGN / "query_index.json").read_text())
valid = {q["id"] for q in roles}
queries = [
    (k, q["query"]) for k, q in enumerate(doc["specific_query_list"]) if f"{meeting}_{k}" in valid
]
suffix = hashlib.sha256(repr((queries, chunks)).encode()).hexdigest()[:12]
enc = ef.Encoder(root)
output = {}
with np.load(root / "embeddings" / f"{meeting}_{suffix}.npz") as z:
    for key, text in [("Q", [q for _, q in queries]), ("C", chunks), ("P", pairs)]:
        error = float(np.max(abs(enc(text) - z[key])))
        assert error < 1e-5
        output[key] = [len(text), error]
(OUTPUT / "encoder_check.json").write_text(
    json.dumps({"meeting": meeting, "device": enc.device, "count_and_max_error": output}, indent=2)
)
print(output)
