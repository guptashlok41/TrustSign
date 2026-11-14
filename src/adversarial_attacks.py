import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import TrafficSignCNN

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"/"gtsrb"
RESULTS=ROOT/"results"
GRAPHS=RESULTS/"graphs"

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILONS=[0.00,0.01,0.03,0.05,0.10]
ITERATIONS=10

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

def fgsm_attack(model,images,labels,epsilon):
    images=images.detach().requires_grad_(True)
    outputs=model(images)
    loss=F.cross_entropy(outputs,labels)
    gradient=torch.autograd.grad(loss,images)[0]
    perturbed=images.detach()+epsilon*gradient.sign()
    return torch.clamp(perturbed,0,1).detach()

def pgd_attack(model,images,labels,epsilon,iterations=10):
    original=images.detach()
    perturbed=original+torch.empty_like(original).uniform_(-epsilon,epsilon)
    perturbed=torch.clamp(perturbed,0,1).detach()

    alpha=epsilon/4

    for _ in range(iterations):
        perturbed=perturbed.detach().requires_grad_(True)
        outputs=model(perturbed)
        loss=F.cross_entropy(outputs,labels)
        gradient=torch.autograd.grad(loss,perturbed)[0]

        perturbed=perturbed.detach()+alpha*gradient.sign()
        delta=torch.clamp(perturbed-original,-epsilon,epsilon)
        perturbed=torch.clamp(original+delta,0,1).detach()

    return perturbed

def evaluate_attack(model,attack,epsilon):
    clean_correct=0
    robust_correct=0
    successful_attacks=0
    total=0

    for images,labels in loader:
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)

        with torch.no_grad():
            clean_predictions=model(images).argmax(dim=1)

        clean_mask=clean_predictions.eq(labels)
        clean_correct+=clean_mask.sum().item()

        if epsilon==0:
            adversarial_images=images
        else:
            adversarial_images=attack(model,images,labels,epsilon)

        with torch.no_grad():
            adversarial_predictions=model(adversarial_images).argmax(dim=1)

        robust_correct+=(adversarial_predictions==labels).sum().item()

        successful_attacks+=(
            clean_mask & adversarial_predictions.ne(labels)
        ).sum().item()

        total+=labels.size(0)

    accuracy=100*robust_correct/total
    asr=100*successful_attacks/clean_correct if clean_correct else 0

    return accuracy,asr,robust_correct,clean_correct

models={
    "Original CNN":"original_cnn.pth",
    "PGD Adversarial CNN":"pgd_adversarial_cnn.pth",
    "Mixed Adversarial CNN":"mixed_adversarial_cnn.pth"
}

all_results=[]

print("Device:",DEVICE)
if DEVICE.type=="cuda":
    print("GPU:",torch.cuda.get_device_name(0))

print("Test images:",len(dataset))

for name,filename in models.items():
    model=TrafficSignCNN().to(DEVICE)
    checkpoint=torch.load(ROOT/"models"/filename,map_location=DEVICE,weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"\n========== {name} ==========")

    for epsilon in EPSILONS:
        fgsm_acc,fgsm_asr,_,_=evaluate_attack(
            model,fgsm_attack,epsilon
        )

        pgd_acc,pgd_asr,_,_=evaluate_attack(
            model,pgd_attack,epsilon
        )

        print(f"\nEpsilon: {epsilon:.2f}")
        print(f"FGSM Accuracy: {fgsm_acc:.2f}%")
        print(f"FGSM ASR: {fgsm_asr:.2f}%")
        print(f"PGD Accuracy: {pgd_acc:.2f}%")
        print(f"PGD ASR: {pgd_asr:.2f}%")

        all_results.append({
            "Model":name,
            "Epsilon":epsilon,
            "FGSM Accuracy":fgsm_acc,
            "FGSM ASR":fgsm_asr,
            "PGD Accuracy":pgd_acc,
            "PGD ASR":pgd_asr
        })

df=pd.DataFrame(all_results)

df.to_csv(RESULTS/"adversarial_results.csv",index=False)

np.savez(
    RESULTS/"adversarial_results.npz",
    results=df.to_records(index=False)
)

for metric,title,filename in [
    ("FGSM Accuracy","FGSM Robust Accuracy","fgsm_accuracy.png"),
    ("PGD Accuracy","PGD Robust Accuracy","pgd_accuracy.png"),
    ("FGSM ASR","FGSM Attack Success Rate","fgsm_asr.png"),
    ("PGD ASR","PGD Attack Success Rate","pgd_asr.png")
]:
    plt.figure(figsize=(9,6))

    for name in models:
        subset=df[df["Model"]==name]
        plt.plot(
            subset["Epsilon"],
            subset[metric],
            marker="o",
            label=name
        )

    plt.xlabel("Epsilon")
    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GRAPHS/filename,dpi=200)
    plt.close()

print("\n========== EXPERIMENT COMPLETE ==========")
print("Results saved:",RESULTS/"adversarial_results.csv")
print("Graphs saved:",GRAPHS)