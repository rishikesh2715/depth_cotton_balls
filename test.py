import os

os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_cache/hub"

from transformers import pipeline
from PIL import Image
import requests

# load pipe
depth_estimator = pipeline(task="depth-estimation", model="Intel/zoedepth-kitti")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
outputs = depth_estimator(image)
depth = outputs["depth"]

print(depth)
