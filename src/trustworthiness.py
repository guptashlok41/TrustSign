
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import TrafficSignCNN

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"/"gtsrb"
RESULTS=ROOT/"results"
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

reports=[]

for name,filename in models.items():
    model=TrafficSignCNN().to(DEVICE)
    checkpoint=torch.load(ROOT/"models"/filename,map_location=DEVICE,weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    correct=0
    incorrect=0
    high_confidence_incorrect=0
    correct_uncertain=0
    high_confidence_correct=0
    needs_verification=0
    confidence_sum=0
    entropy_sum=0
    total=0

    with torch.no_grad():
        for images,labels in loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)

            outputs=model(images)
            probabilities=F.softmax(outputs,dim=1)

            confidence,predictions=probabilities.max(dim=1)
            entropy=-(probabilities*torch.log(probabilities+1e-10)).sum(dim=1)
            normalized_entropy=entropy/np.log(43)

            matches=predictions.eq(labels)

            correct+=matches.sum().item()
            incorrect+=(~matches).sum().item()

            confidence_sum+=confidence.sum().item()
            entropy_sum+=normalized_entropy.sum().item()

            high_confidence_incorrect+=(
                (~matches)&(confidence>=0.90)
            ).sum().item()

            correct_uncertain+=(
                matches&((confidence<0.60)|(normalized_entropy>0.60))
            ).sum().item()

            high_confidence_correct+=(
                matches&(confidence>=0.90)&(normalized_entropy<=0.20)
            ).sum().item()

            needs_verification+=(
                matches&~(
                    ((confidence<0.60)|(normalized_entropy>0.60))|
                    ((confidence>=0.90)&(normalized_entropy<=0.20))
                )
            ).sum().item()

            total+=labels.size(0)

    accuracy=100*correct/total
    avg_confidence=100*confidence_sum/total
    avg_entropy=entropy_sum/total

    report={
        "Model":name,
        "Accuracy (%)":round(accuracy,2),
        "Average Confidence (%)":round(avg_confidence,2),
        "Average Normalized Entropy":round(avg_entropy,4),
        "Correct":correct,
        "Incorrect":incorrect,
        "High Confidence Incorrect":high_confidence_incorrect,
        "Correct but Uncertain":correct_uncertain,
        "High Confidence Correct":high_confidence_correct,
        "Needs Further Verification":needs_verification
    }

    reports.append(report)

    print(f"\n========== {name} ==========")
    for key,value in report.items():
        if key!="Model":
            print(f"{key}: {value}")

df=pd.DataFrame(reports)
df.to_csv(RESULTS/"trustworthiness_report_local.csv",index=False)

print("\n========== ANALYSIS COMPLETE ==========")
print(df.to_string(index=False))
print("\nSaved:",RESULTS/"trustworthiness_report_local.csv")


