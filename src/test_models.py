import torch
from pathlib import Path
from model import TrafficSignCNN

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_dir=Path(__file__).resolve().parent.parent

models={
    "Original CNN":"original_cnn.pth",
    "PGD Adversarial CNN":"pgd_adversarial_cnn.pth",
    "Mixed Adversarial CNN":"mixed_adversarial_cnn.pth"
}

print("Device:",device)
print("GPU:",torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")

for name,filename in models.items():
    path=project_dir/"models"/filename
    model=TrafficSignCNN().to(device)
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    dummy_input=torch.randn(1,3,32,32).to(device)

    with torch.no_grad():
        output=model(dummy_input)
        probabilities=torch.softmax(output,dim=1)
        prediction=probabilities.argmax(dim=1).item()
        confidence=probabilities.max().item()*100

    print(f"\n{name}")
    print("Loaded successfully")
    print("Parameters:",sum(p.numel() for p in model.parameters()))
    print("Device:",next(model.parameters()).device)
    print("Output shape:",output.shape)
    print("Predicted class:",prediction)
    print(f"Confidence: {confidence:.2f}%")