import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from prepare_data_cityscapes import cityscapesLoader

def get_bounding_box(mask):
    """Generate bounding box (x_min, y_min, x_max, y_max) for a given mask."""
    # Find the non-zero regions (which belong to the class)
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None  # Return None if there is no segmentation for the class
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

def generate_annotations(loader, split):
    """Generates annotations.json for the dataset."""
    annotations = {}
    
    # Loop over the dataset and process each image and label
    for idx, (image, label) in tqdm(enumerate(loader), total=len(loader), desc=f"Processing {split} set"):
        filename = loader.dataset.files[split][idx].split(os.sep)[-1]  # Get the filename
        # Path to image and mask
        image_path = loader.dataset.files[split][idx]
        mask_path = image_path.replace("leftImg8bit", "gtFine").replace(".png", "_gtFine_labelIds.png")

        bboxes = []

        # Iterate over each class (0-18) and generate bounding boxes
        for class_id in range(19):
            # Create a binary mask for the current class
            mask = (label[0].cpu().numpy() == class_id).astype(np.uint8)
            bbox = get_bounding_box(mask)
            if bbox:
                bboxes.append({
                    "class_id": class_id,
                    "bbox": bbox
                })

        # Store image and mask paths and bounding boxes in the annotations
        annotations[filename] = {
            "image_path": image_path,
            "mask_path": mask_path,
            "bboxes": bboxes  # List of bounding boxes for each class in the image
        }

    return annotations

def save_annotations_to_json(annotations, output_path):
    """Save the annotations dictionary to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    ROOT_PATH = "/tmp/ahsan/sqfs/storage_local/datasets/public/cityscapes/"
    path_data = ROOT_PATH
    batch_size = 1
    num_workers = 1
    annotations_path = 'annotations_cityscapes.json'
    
    # Initialize the dataset and dataloader for the train split
    train_data = cityscapesLoader(root=path_data, split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Process the training dataset and generate annotations
    train_annotations = generate_annotations(train_loader, 'train')

    # Initialize the dataset and dataloader for the val split
    val_data = cityscapesLoader(root=path_data, split='val')
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Process the validation dataset and generate annotations
    val_annotations = generate_annotations(val_loader, 'val')

    # Combine train and val annotations
    annotations = {
        "train": train_annotations,
        "val": val_annotations
    }

    # Save the annotations to a JSON file
    save_annotations_to_json(annotations, annotations_path)

    print(f"Annotations saved to {annotations_path}")
