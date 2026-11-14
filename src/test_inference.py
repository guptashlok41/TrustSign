from inference import load_models,predict_image,generate_gradcam
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent

models=load_models()

image_path=ROOT/"data"/"gtsrb"/"Test"/"00000.png"
test=pd.read_csv(ROOT/"data"/"gtsrb"/"Test.csv")
actual_class=int(test.iloc[0]["ClassId"])

image,tensor,results=predict_image(models,image_path,actual_class)

output_path=ROOT/"results"/"graphs"/"gradcam_test.png"

generate_gradcam(models,image,tensor,output_path)