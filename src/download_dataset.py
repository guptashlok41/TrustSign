import kagglehub
import shutil
from pathlib import Path

path=kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

destination=Path(__file__).resolve().parent.parent/"data"/"gtsrb"

shutil.copytree(path,destination,dirs_exist_ok=True)

print("Dataset downloaded successfully!")
print("Location:",destination)