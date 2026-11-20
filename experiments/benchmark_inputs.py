"""External text and local model adapters for optional neural benchmarks."""
from pathlib import Path
import json
import zipfile
import numpy as np
from .paths import DATA, AUDIT


def title_records():
    queries = json.loads((DATA / "forum" / "body_rerun_queries.json").read_text())
    with np.load(AUDIT / "forum.npz") as data:
        values = data["values"]
    assert len(queries) == len(values)
    return [
        dict(
            domain=q["forum"],
            query=q["id"],
            positive=q["positive"],
            p=v[:, 2],
            y=v[:, 3],
            arms={"0": {"p": q["center"]}, "1": {"p": q["protected"], "pairs": q["probes"]}},
        )
        for q, v in zip(queries, values)
    ]


class CrossEncoder:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.net = (
            AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            .to(self.device)
            .eval()
        )


def design_timing_inputs(archive):
    stored = json.loads((DATA / "timing" / "design_readouts.json").read_text())
    workloads = {}
    with zipfile.ZipFile(archive) as source:
        for split in ("val", "test"):
            rows = []
            for path in sorted(source.namelist()):
                if f"/Product/{split}/" not in path or not path.endswith(".json"):
                    continue
                meeting = Path(path).stem
                doc = json.loads(source.read(path))
                turns = [x["speaker"] + ": " + x["content"] for x in doc["meeting_transcripts"]]
                chunks = [" ".join(turns[i : i + 8]) for i in range(0, len(turns), 8)]
                for i, q in enumerate(doc["specific_query_list"]):
                    uid = f"{meeting}_{i}"
                    if uid in stored[split]:
                        s = stored[split][uid]
                        rows.append(
                            (
                                q["query"],
                                chunks,
                                np.array(s["r"]),
                                np.array(s["b"]),
                                np.array(s["p"]),
                                uid,
                            )
                        )
            assert len(rows) == len(stored[split])
            workloads[split] = rows
    return workloads["test"], workloads["val"], stored["logit_slope"], stored["logit_bias"]
