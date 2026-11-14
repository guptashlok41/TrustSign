import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import TrafficSignCNN

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"/"gtsrb"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GTSRBDataset(Dataset):
    def __init__(self,csv_file,root,transform=None):
        self.data=pd.read_csv(csv_file)
        self.root=root
        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        row=self.data.iloc[index]
        image=Image.open(self.root/row["Path"]).convert("RGB")
        label=int(row["ClassId"])

        if self.transform:
            image=self.transform(image)

        return image,label

transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

dataset=GTSRBDataset(DATA/"Test.csv",DATA,transform)
loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,pin_memory=DEVICE.type=="cuda")

models={
    "Original CNN":"original_cnn.pth",
    "PGD Adversarial CNN":"pgd_adversarial_cnn.pth",
    "Mixed Adversarial CNN":"mixed_adversarial_cnn.pth"
}

print("Device:",DEVICE)
print("GPU:",torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "CPU")
print("Test images:",len(dataset))

for name,filename in models.items():
    model=TrafficSignCNN().to(DEVICE)
    checkpoint=torch.load(ROOT/"models"/filename,map_location=DEVICE,weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    correct=0
    total=0

    with torch.no_grad():
        for images,labels in loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)

            outputs=model(images)
            predictions=outputs.argmax(dim=1)

            correct+=(predictions==labels).sum().item()
            total+=labels.size(0)

    accuracy=100*correct/total

    print(f"\n{name}")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Test Accuracy: {accuracy:.2f}%")