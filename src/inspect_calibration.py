import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

files=[
    ROOT/"results"/"final_results.json",
    ROOT/"results"/"TrustSign_Complete_Results.json"
]

for path in files:
    print(f"\n========== {path.name} ==========")

    with open(path,"r") as f:
        data=json.load(f)

    print(json.dumps(data,indent=4))