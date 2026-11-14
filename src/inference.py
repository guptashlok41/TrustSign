
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from model import TrafficSignCNN

ROOT=Path(__file__).resolve().parent.parent
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

CLASS_NAMES={
    0:"Speed Limit 20",
    1:"Speed Limit 30",
    2:"Speed Limit 50",
    3:"Speed Limit 60",
    4:"Speed Limit 70",
    5:"Speed Limit 80",
    6:"End of Speed Limit 80",
    7:"Speed Limit 100",
    8:"Speed Limit 120",
    9:"No Passing",
    10:"No Passing for Vehicles Over 3.5 Tons",
    11:"Right-of-Way at Next Intersection",
    12:"Priority Road",
    13:"Yield",
    14:"Stop",
    15:"No Vehicles",
    16:"Vehicles Over 3.5 Tons Prohibited",
    17:"No Entry",
    18:"General Caution",
    19:"Dangerous Curve to the Left",
    20:"Dangerous Curve to the Right",
    21:"Double Curve",
    22:"Bumpy Road",
    23:"Slippery Road",
    24:"Road Narrows on the Right",
    25:"Road Work",
    26:"Traffic Signals",
    27:"Pedestrians",
    28:"Children Crossing",
    29:"Bicycles Crossing",
    30:"Beware of Ice/Snow",
    31:"Wild Animals Crossing",
    32:"End of All Speed and Passing Limits",
    33:"Turn Right Ahead",
    34:"Turn Left Ahead",
    35:"Ahead Only",
    36:"Go Straight or Right",
    37:"Go Straight or Left",
    38:"Keep Right",
    39:"Keep Left",
    40:"Roundabout Mandatory",
    41:"End of No Passing",
    42:"End of No Passing by Vehicles Over 3.5 Tons"
}

MODELS={
    "Original CNN":"original_cnn.pth",
    "PGD Adversarial CNN":"pgd_adversarial_cnn.pth",
    "Mixed Adversarial CNN":"mixed_adversarial_cnn.pth"
}

def load_models():
    loaded={}

    for name,filename in MODELS.items():
        model=TrafficSignCNN().to(DEVICE)
        checkpoint=torch.load(
            ROOT/"models"/filename,
            map_location=DEVICE,
            weights_only=True
        )
        model.load_state_dict(checkpoint)
        model.eval()
        loaded[name]=model

    return loaded

def preprocess_image(image_path):
    image=Image.open(image_path).convert("RGB")
    tensor=TRANSFORM(image).unsqueeze(0).to(DEVICE)
    return image,tensor

def predict_image(models,image_path,actual_class=None):
    image,tensor=preprocess_image(image_path)

    print("\n========== PREDICTION RESULTS ==========")
    print("Device:",DEVICE)

    results={}

    for name,model in models.items():
        with torch.no_grad():
            output=model(tensor)
            probabilities=F.softmax(output,dim=1)

            confidence,prediction=probabilities.max(dim=1)

            entropy=-(
                probabilities*torch.log(probabilities+1e-10)
            ).sum(dim=1)

            normalized_entropy=(
                entropy/np.log(43)
            ).item()

        prediction=prediction.item()
        confidence=confidence.item()
        sign_name=CLASS_NAMES[prediction]

        results[name]={
            "prediction":prediction,
            "class_name":sign_name,
            "confidence":confidence,
            "entropy":normalized_entropy
        }

        print(f"\n{name}")
        print("Predicted Sign:",sign_name)
        print("Class ID:",prediction)
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Normalized Entropy: {normalized_entropy:.4f}")

        if actual_class is not None:
            print("Actual Sign:",CLASS_NAMES.get(actual_class,"Unknown"))
            print("Correct:",prediction==actual_class)

    return image,tensor,results

def generate_gradcam(models,image,tensor,output_path):
    fig,axes=plt.subplots(1,4,figsize=(16,4))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for index,(name,model) in enumerate(models.items(),start=1):
        activations={}

        def save_activation(module,inputs,output):
            output.retain_grad()
            activations["value"]=output

        hook=model.conv3.register_forward_hook(save_activation)

        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output=model(tensor)
            prediction=output.argmax(dim=1).item()
            score=output[0,prediction]
            score.backward()

        feature_map=activations["value"]
        gradients=feature_map.grad

        if gradients is None:
            hook.remove()
            raise RuntimeError(f"Gradients not captured for {name}")

        weights=gradients.mean(dim=(2,3),keepdim=True)
        cam=(weights*feature_map).sum(dim=1,keepdim=True)
        cam=F.relu(cam)

        cam=F.interpolate(
            cam,
            size=(32,32),
            mode="bilinear",
            align_corners=False
        )

        heatmap=cam.squeeze().detach().cpu().numpy()
        heatmap=heatmap/(heatmap.max()+1e-8)

        sign_name=CLASS_NAMES[prediction]

        axes[index].imshow(image)
        axes[index].imshow(heatmap,cmap="jet",alpha=0.5)
        axes[index].set_title(f"{name}\n{sign_name} ({prediction})")
        axes[index].axis("off")

        hook.remove()

    plt.tight_layout()
    plt.savefig(output_path,dpi=200,bbox_inches="tight")
    plt.close(fig)

    print("\nGrad-CAM saved:",output_path)
