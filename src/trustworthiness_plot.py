
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
RESULTS=ROOT/"results"

df=pd.read_csv(RESULTS/"trustworthiness_report_local.csv")

df["Low Confidence Incorrect"]=(
    df["Incorrect"]-df["High Confidence Incorrect"]
)

metrics=[
    "High Confidence Incorrect",
    "Low Confidence Incorrect",
    "Correct but Uncertain",
    "High Confidence Correct",
    "Needs Further Verification"
]

totals=(
    df["High Confidence Incorrect"]+
    df["Low Confidence Incorrect"]+
    df["Correct but Uncertain"]+
    df["High Confidence Correct"]+
    df["Needs Further Verification"]
)

print("Prediction category totals:")
for model,total in zip(df["Model"],totals):
    print(f"{model}: {total}/12630")

if not (totals==12630).all():
    raise ValueError("Prediction categories do not sum to 12630!")

ax=df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(13,7),
    width=0.8
)

plt.title("TrustSign - Complete Trustworthiness Comparison")
plt.ylabel("Number of Predictions")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.02,1),loc="upper left")
plt.grid(axis="y",alpha=0.3)
plt.tight_layout()

path=RESULTS/"graphs"/"trustworthiness_categories.png"
plt.savefig(path,dpi=200,bbox_inches="tight")
plt.show()

print("\nGraph saved:",path)

