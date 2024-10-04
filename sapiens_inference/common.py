import os
import shutil
from typing import List
import requests
from tqdm import tqdm
from enum import Enum
from huggingface_hub import hf_hub_download, hf_hub_url

from torchvision import transforms


class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"


def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)

def download_hf_model(model_filename: str, task: TaskType) -> str:
    # Define HuggingFace direct download URLs based on model types
    hf_links = {
        "sapiens_0.3b_render_people_epoch_100_torchscript.pt2": "https://huggingface.co/facebook/sapiens-depth-03b/resolve/main/sapiens_0.3b_depth_render_people_epoch_100.pth",
        "sapiens_0.6b_render_people_epoch_70_torchscript.pt2": "https://huggingface.co/facebook/sapiens-depth-06b/resolve/main/sapiens_0.6b_depth_render_people_epoch_70.pth",
        "sapiens_1b_render_people_epoch_88_torchscript.pt2": "https://huggingface.co/facebook/sapiens-normal-1b/resolve/main/sapiens_1b_normal_render_people_epoch_115.pth",
        "sapiens_2b_render_people_epoch_25_torchscript.pt2": "https://huggingface.co/facebook/sapiens-depth-2b/resolve/main/sapiens_2b_depth_render_people_epoch_25.pth"
    }
    
    url = hf_links.get(model_filename)
    if not url:
        raise ValueError(f"No URL found for model {model_filename}")
    
    # Download the model
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)

    if not os.path.exists(model_path):
        print(f"Downloading {model_filename} from {url}...")
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as model_file:
            model_file.write(response.content)
        print(f"Downloaded {model_filename} to {model_path}")
    else:
        print(f"Model {model_filename} already exists at {model_path}")
    
    return model_path



def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               transforms.Lambda(lambda x: x.unsqueeze(0))
                               ])
