import torch
import numpy as np 
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils
from PIL import Image, ImageDraw
import yaml
import json
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='9'

"""
This file is used to plot the predictions of a model (either baseline or LoRA) on the train or test set. Most of it is hard coded so I would like to explain some parameters to change 
referencing by lines : 
line 22: change the rank of lora; line 98: Do inference on train (inference_train=True) else on test; line 101 and 111 is_baseline arguments in the function: True to use baseline False to use LoRA model. 
"""
sam_checkpoint = "sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = build_sam_vit_b(checkpoint=sam_checkpoint)
rank = 512
sam_lora = LoRA_sam(sam, rank)
sam_lora.load_lora_parameters(f"./lora_weights/lora_rank{rank}.safetensors")
model = sam_lora.sam

def inference_model(sam_model, image_path, filename, mask_path=None, bbox=None, is_baseline=False):
    if is_baseline == False:
        model = sam_model.sam
        rank = sam_model.rank
    else:
        model = build_sam_vit_b(checkpoint=sam_checkpoint)

    model.eval()
    model.to(device)
    image = Image.open(image_path)
    if mask_path is not None:
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        ground_truth_mask = np.array(mask)
        ground_truth_box = utils.get_bounding_box(ground_truth_mask)
    else:
        ground_truth_box = bbox

    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(ground_truth_box),
        multimask_output=False,
    )

    predicted_box = utils.get_bounding_box(masks[0])

    # Plotting the image with both ground truth and predicted bounding boxes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    draw = ImageDraw.Draw(image)
    draw.rectangle(ground_truth_box, outline="red", width=3)  # Ground truth box in red
    draw.rectangle(predicted_box, outline="blue", width=3)    # Predicted box in blue

    ax.imshow(image)
    ax.set_title(f"Image with Ground Truth (Red) and Predicted (Blue) Bounding Boxes: {filename}")
    plt.savefig(f"./plots/{filename}_comparison.jpg")

    if mask_path is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
        ax1.imshow(image)
        ax1.set_title(f"Image with Bounding Boxes: {filename}")

        ax2.imshow(ground_truth_mask)
        ax2.set_title(f"Ground truth mask: {filename}")

        ax3.imshow(masks[0])
        if is_baseline:
            ax3.set_title(f"Baseline SAM prediction: {filename}")
            plt.savefig(f"./plots/{filename}_baseline.jpg")
        else:
            ax3.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
            plt.savefig(f"./plots/{filename[:-4]}_rank{rank}.jpg")

# Open configuration file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)

# Open annotation file
f = open('annotations.json')
annotations = json.load(f)

train_set = annotations["train"]
test_set = annotations["test"]
inference_train = True

if inference_train:
    for image_name, value in tqdm(train_set.items()):
        torch.cuda.empty_cache()
        image_path = value.get("image_path")
        mask_path = value.get("mask_path")
        bbox = value.get("bbox")
        inference_model(sam_lora, image_path, filename=image_name, mask_path=mask_path, bbox=bbox, is_baseline=False)
else:
    for image_name, value in test_set.items():
        torch.cuda.empty_cache()
        image_path = value.get("image_path")
        mask_path = value.get("mask_path")
        bbox = value.get("bbox")
        inference_model(sam_lora, image_path, filename=image_name, mask_path=mask_path, bbox=bbox, is_baseline=False)
        
