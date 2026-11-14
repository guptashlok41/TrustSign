
import sys
import json
import pandas as pd
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT/"src"

sys.path.insert(0,str(SRC))

from inference import load_models,predict_image,preprocess_image,generate_gradcam

MODELS=None

def get_models():
    global MODELS
    if MODELS is None:
        print("\nLoading CNN models...")
        MODELS=load_models()
        print("All models loaded successfully!")
    return MODELS

def print_header():
    print("\n"+"="*55)
    print("                    TRUSTSIGN")
    print("       Trustworthy Traffic Sign Recognition")
    print("="*55)

def get_image_path():
    path=input("\nEnter image path: ").strip().strip('"')
    image_path=Path(path)

    if not image_path.is_absolute():
        image_path=ROOT/image_path

    if not image_path.exists():
        print("Error: Image file not found!")
        return None

    return image_path

def prediction():
    image_path=get_image_path()
    if image_path is None:
        return

    actual=input("Enter actual class ID (press Enter to skip): ").strip()

    if actual:
        try:
            actual=int(actual)
            if not 0<=actual<43:
                print("Class ID must be between 0 and 42.")
                return
        except ValueError:
            print("Invalid class ID.")
            return
    else:
        actual=None

    models=get_models()

    try:
        _,_,results=predict_image(models,image_path,actual)
    except Exception as e:
        print("Prediction failed:",e)

def gradcam():
    image_path=get_image_path()
    if image_path is None:
        return

    try:
        models=get_models()
        image,tensor=preprocess_image(image_path)

        output_path=ROOT/"results"/"graphs"/"gradcam_custom.png"

        generate_gradcam(
            models,
            image,
            tensor,
            output_path
        )

        print("Open the saved image to view the Grad-CAM comparison.")

    except Exception as e:
        print("Grad-CAM failed:",e)

def adversarial_results():
    path=ROOT/"results"/"adversarial_results.csv"

    if not path.exists():
        print("Adversarial results CSV not found.")
        return

    df=pd.read_csv(path)

    print("\n========== ADVERSARIAL ROBUSTNESS RESULTS ==========")
    print(df.to_string(index=False))

def trustworthiness_results():
    path=ROOT/"results"/"trustworthiness_report_local.csv"

    if not path.exists():
        print("Trustworthiness report not found.")
        return

    df=pd.read_csv(path)

    print("\n========== TRUSTWORTHINESS ANALYSIS ==========")
    print(df.to_string(index=False))

def calibration_results():
    path=ROOT/"results"/"calibration_results.json"

    if not path.exists():
        print("Calibration results not found.")
        return

    with open(path,"r") as file:
        data=json.load(file)

    print("\n========== CALIBRATION ANALYSIS ==========")

    for key,value in data.items():
        print(f"{key}: {value}")

def overall_results():
    path=ROOT/"results"/"final_results.json"

    if not path.exists():
        print("Final results JSON not found.")
        return

    with open(path,"r") as file:
        data=json.load(file)

    print("\n========== OVERALL MODEL COMPARISON ==========")

    for model,metrics in data.items():
        print(f"\n{model}")
        if isinstance(metrics,dict):
            for metric,value in metrics.items():
                print(f"  {metric}: {value}")

def main():
    while True:
        print_header()

        print("1. Predict Traffic Sign")
        print("2. Grad-CAM Explainability")
        print("3. Adversarial Robustness Results")
        print("4. Trustworthiness Analysis")
        print("5. Calibration Analysis")
        print("6. Overall Experimental Results")
        print("7. Exit")

        choice=input("\nEnter your choice: ").strip()

        if choice=="1":
            prediction()

        elif choice=="2":
            gradcam()

        elif choice=="3":
            adversarial_results()

        elif choice=="4":
            trustworthiness_results()

        elif choice=="5":
            calibration_results()

        elif choice=="6":
            overall_results()

        elif choice=="7":
            print("\nThank you for using TrustSign!")
            print("Exiting application...")
            break

        else:
            print("\nInvalid choice! Please select 1-7.")

        input("\nPress Enter to return to the main menu...")

if __name__=="__main__":
    main()

