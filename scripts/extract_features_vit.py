# source: https://huggingface.co/trpakov/vit-face-expression

import sys
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

# 'trpakov/vit-face-expression'
model_card = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_card)
model = ViTModel.from_pretrained(model_card)

# demo
if 1:
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape) # torch.Size([1, 197, 768])
    sys.exit(0)

# muse provided vit