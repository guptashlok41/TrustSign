
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

image_path=ROOT/"data"/"gtsrb"/"Test"/"00000.png"
image=Image.open(image_path).convert("RGB").resize((32,32))
tensor=transform(image).unsqueeze(0).to(DEVICE)

models={
    "Original CNN":"original_cnn.pth",
    "PGD Adversarial CNN":"pgd_adversarial_cnn.pth",
    "Mixed Adversarial CNN":"mixed_adversarial_cnn.pth"
}

fig,axes=plt.subplots(1,4,figsize=(16,4))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

for index,(name,filename) in enumerate(models.items(),start=1):
    model=TrafficSignCNN().to(DEVICE)
    checkpoint=torch.load(ROOT/"models"/filename,map_location=DEVICE,weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    activations={}

    def save_activation(module,input,output):
        output.retain_grad()
        activations["value"]=output

    hook=model.conv3.register_forward_hook(save_activation)

    model.zero_grad(set_to_none=True)

    output=model(tensor)
    prediction=output.argmax(dim=1).item()
    score=output[0,prediction]
    score.backward()

    feature_map=activations["value"]
    gradients=feature_map.grad

    if gradients is None:
        raise RuntimeError(f"Gradients not captured for {name}")

    weights=gradients.mean(dim=(2,3),keepdim=True)
    cam=(weights*feature_map).sum(dim=1,keepdim=True)
    cam=F.relu(cam)
    cam=F.interpolate(cam,size=(32,32),mode="bilinear",align_corners=False)

    heatmap=cam.squeeze().detach().cpu().numpy()
    heatmap=heatmap/(heatmap.max()+1e-8)

    axes[index].imshow(image)
    axes[index].imshow(heatmap,cmap="jet",alpha=0.5)
    axes[index].set_title(f"{name}\nClass: {prediction}")
    axes[index].axis("off")

    hook.remove()

plt.tight_layout()

output_path=ROOT/"results"/"graphs"/"gradcam_comparison.png"
plt.savefig(output_path,dpi=200,bbox_inches="tight")
plt.show()

print("Grad-CAM comparison saved:",output_path)

