import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
import json
import yaml
from typing import Literal
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UFPR_ALPR_Dataset:
    def __init__(self, root, split="training"):
        self.split = split
        self.data_dir = os.path.join(root, self.split)
        self.image_list = self.build_image_list()
        logger.info(f"Dataset initialized with {len(self.image_list)} images.")

    def build_image_list(self):
        image_list = []
        for i in range(len(os.listdir(self.data_dir))):
            path = os.path.join(self.data_dir, os.listdir(self.data_dir)[i])
            files = os.listdir(path)
            for j in range(len(files)):
                if os.path.splitext(files[j])[-1] == ".png":
                    image_list.append(os.path.join(path, files[j]))
        return image_list

    def load_image(self, path):
        img = Image.open(path).convert("RGB")  # Convert to RGB
        img = np.array(img, dtype=np.uint8)
        return img

    def load_annotations(self, path):
        file = path.replace("png", "txt")
        with open(file, "r") as f:
            data = f.read()
        coordinates = data.splitlines()[7].split(":")[1].strip().split(" ")
        corners = []
        for coord in coordinates:
            x, y = coord.split(',')
            corners.append((int(x), int(y)))
        corners = np.array(corners, dtype=np.float32)
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])

        label = np.array([x_min, y_min, x_max, y_max])
        label = label.reshape((1, 4))
        return label

    def plate_mask(self, img, annot):
        h, w = img.shape[0], img.shape[1]
        mask = np.zeros((h, w), dtype=np.uint8)

        x_min, y_min, x_max, y_max = annot[0]

        mask[int(y_min):int(y_max), int(x_min):int(x_max)] = 255

        return mask

    def save_image(self, img, save_path):
        img_pil = Image.fromarray(img)
        img_pil.save(save_path)

    def save_mask(self, mask, save_path):
        mask_pil = Image.fromarray(mask)
        mask_pil.save(save_path)

    def prepare_data(self):
        dataset_annotations = {}

        save_image_dir = os.path.join(os.getcwd(), "ufpr_dataset", self.split, "images")
        save_mask_dir = os.path.join(os.getcwd(), "ufpr_dataset", self.split, "masks")
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_mask_dir, exist_ok=True)
        
        for idx, path in tqdm(enumerate(self.image_list, start=0), desc="Data Loading in process"):
            try:
                img = self.load_image(path)
                filename = os.path.basename(path)
                image_path = os.path.join(save_image_dir, filename)
                # self.save_image(img=img, save_path=image_path)

                plate_annot = self.load_annotations(path)
                mask = self.plate_mask(img, plate_annot)
                mask_path = os.path.join(save_mask_dir, filename)
                # self.save_mask(mask=mask, save_path=mask_path)

                dataset_annotations[filename] = {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "bbox": plate_annot[0].tolist()  # Convert numpy array to list for JSON serialization
                }

            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        
        logger.info("Data preparation completed.")
        return dataset_annotations

def prepare_annotations(dataset_path:str,split:Literal['training','testing']):
    
    annotations_file_name="_".join("20241111_105827_config.yaml".split("_")[0:-1])+"_annotations.json"



    if os.path.exists(annotations_file_name):
        with open(annotations_file_name, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {"train": {}, "test": {}}

    for split in ["training", "testing"]:
        logging.info(f"Preparing annotations for {split}")

        dataset = UFPR_ALPR_Dataset(root=dataset_path, split=split)
        data = dataset.prepare_data()

        if split == "training":
            annotations["train"] = data
        else:
            annotations["test"] = data

    with open(annotations_file_name, 'w') as f: 
        json.dump(annotations, f, indent=4)
