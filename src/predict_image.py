import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from torchvision import transforms
from PIL import Image
from model import TrafficSignCNN

ROOT=Path(__file__).resolve().parent.parent
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

models={
    "Original CNN":"original_cnn.pth",
    "PGD Adversarial CNN":"pgd_adversarial_cnn.pth",
    "Mixed Adversarial CNN":"mixed_adversarial_cnn.pth"
}

image_path=ROOT/"data"/"gtsrb"/"Test"/"00000.png"

test=pd.read_csv(ROOT/"data"/"gtsrb"/"Test.csv")
actual_class=int(test.iloc[0]["ClassId"])

image=Image.open(image_path).convert("RGB")
image_tensor=transform(image).unsqueeze(0).to(DEVICE)

print("Actual Class:",actual_class)
print("Device:",DEVICE)

for name,filename in models.items():
    model=TrafficSignCNN().to(DEVICE)
    checkpoint=torch.load(ROOT/"models"/filename,map_location=DEVICE,weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        output=model(image_tensor)
        probabilities=F.softmax(output,dim=1)
        prediction=probabilities.argmax(dim=1).item()
        confidence=probabilities.max().item()
        entropy=-(probabilities*torch.log(probabilities+1e-10)).sum(dim=1)
        normalized_entropy=(entropy/torch.log(torch.tensor(43.0,device=DEVICE))).item()

    print(f"\n{name}")
    print("Predicted Class:",prediction)
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Normalized Entropy: {normalized_entropy:.4f}")
    print("Correct:",prediction==actual_class)