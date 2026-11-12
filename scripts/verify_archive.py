"""Verify archived bytes and candidate summary, without training or private data."""
from pathlib import Path
import csv, hashlib, json
R=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((R/'results/provenance/source_manifest.json').read_text())
    for r in manifest:
        assert hashlib.sha256((R/r['destination']).read_bytes()).hexdigest()==r['sha256'],r['destination']
    rows=list(csv.DictReader((R/'results/summary/candidate_results.csv').open()))
    assert len(rows)==6
    for r in rows:
        assert abs(float(r['mtl_mae'])-float(r['candidate_mae'])-float(r['mtl_gain']))<1e-12
        assert float(r['mtl_gain'])>0 and float(r['mtl_low'])<0<float(r['mtl_high'])
    receipts=json.loads((R/'results/provenance/stage151_receipts.json').read_text())
    assert len(receipts)==90 and sum(r['fits'] for r in receipts)==2520
    print(f'PASS: {len(manifest)} exact source artifacts; 90 receipt records; six summary settings.')
if __name__=='__main__':main()
