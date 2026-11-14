import pandas as pd
from pathlib import Path

root=Path(__file__).resolve().parent.parent/"data"/"gtsrb"

train=pd.read_csv(root/"Train.csv")
test=pd.read_csv(root/"Test.csv")
meta=pd.read_csv(root/"Meta.csv")

print("========== DATASET OVERVIEW ==========")
print("Training images:",len(train))
print("Testing images:",len(test))
print("Metadata records:",len(meta))

print("\n========== COLUMNS ==========")
print("Train:",train.columns.tolist())
print("Test:",test.columns.tolist())

print("\n========== CLASS DISTRIBUTION ==========")
print("Training classes:",train["ClassId"].nunique())
print("Testing classes:",test["ClassId"].nunique())

print("\n========== SAMPLE RECORDS ==========")
print(train.head(3).to_string(index=False))
print("\nTest sample:")
print(test.head(3).to_string(index=False))

print("\n========== IMAGE PATH CHECK ==========")
for relative_path in train["Path"].head(5):
    print(relative_path,"->",(root/relative_path).exists())