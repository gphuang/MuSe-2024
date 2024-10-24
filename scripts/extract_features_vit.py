# source: https://huggingface.co/trpakov/vit-face-expression
# facial landmarks: https://github.com/1adrianb/face-alignment?tab=readme-ov-file
# face & facial landmarks detection: https://ai.google.dev/edge/mediapipe/solutions/guide
# muse mtcnn: https://github.com/ipazc/mtcnn
# source activate ser_venv

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
    import cv2
    # from PIL import Image 
    from mtcnn import MTCNN
    from mtcnn.utils.images import load_image
    from mtcnn.utils.plotting import plot
    import matplotlib.pyplot as plt
    IMAGE_FILE = '/scratch/work/huangg5/tutorials/data_example/image.jpg'
    OUT_FNAME = '/scratch/work/huangg5/tutorials/data_example/savedImage.jpg'
    detector = MTCNN(device="CPU:0")
    # image = load_image(IMAGE_FILE)
    image = cv2.imread(IMAGE_FILE) # RGB color order flipped
    result_list = detector.detect_faces(image)
    image_w_box = plot(image, result_list)
    #plt.imsave(OUT_FNAME, image_w_box)
    cv2.imwrite(OUT_FNAME, image_w_box)
    # save each box
    for i, result in enumerate(result_list):
        x, y, width, height = result['box']
        # extract the face
        roi_face = image[y:y+height, x:x+width]
        # extract detailed landmarks
        # for key, value in result['keypoints'].items():
        # plt.imsave(f"/scratch/work/huangg5/tutorials/data_example/roi_{i}.jpg", roi_face)
        cv2.imwrite(f"/scratch/work/huangg5/tutorials/data_example/roi_{i}.jpg", roi_face)
    sys.exit(0)

    # extract vit from face
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    image = load_image(IMAGE_FILE) # OUT_FNAME
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape) # torch.Size([1, 197, 768])
    sys.exit(0)



# muse provided vit