import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class cityscapesLoader(data.Dataset):
    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    # Map each class to its corresponding color
    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        root,
        split="train",
        is_transform=True,
        img_size=(512, 1024),
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7, 8, 11, 12, 13, 17, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # Path of image
        img_path = self.files[self.split][index].rstrip()

        # Path of label
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        # Read image using OpenCV
        img = cv2.imread(img_path)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read label using OpenCV
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        # Encode labels
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        # Resize image
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW

        # Resize label
        lbl = cv2.resize(lbl, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        classes = np.unique(lbl)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        # Convert to PyTorch tensors
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
      
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Encode segmentation map
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def save_image(self, img, save_path):
        # Convert tensor to numpy array and transpose to HWC
        img_np = img.numpy().transpose(1, 2, 0).astype(np.uint8)
        # Convert from RGB to BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Save image
        cv2.imwrite(save_path, img_np)

    def save_label(self, lbl, save_path):
        # Convert tensor to numpy array
        lbl_np = lbl.numpy().astype(np.uint8)
        # Save label as PNG
        lbl_pil = Image.fromarray(lbl_np, mode='L')
        lbl_pil.save(save_path)

    def prepare_data(self):
        save_image_dir = os.path.join(os.getcwd(), "cityscapes_dataset", self.split, "images")
        save_label_dir = os.path.join(os.getcwd(), "cityscapes_dataset", self.split, "masks")
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_label_dir, exist_ok=True)

        for idx in tqdm(range(len(self.files[self.split])), desc=f"Processing Cityscapes {self.split} data"):
            img_path = self.files[self.split][idx].rstrip()
            try:
                img, lbl = self.__getitem__(idx)

                # Save image
                filename = os.path.basename(img_path)
                image_save_path = os.path.join(save_image_dir, filename)
                self.save_image(img, image_save_path)

                # Save label
                label_filename = filename.replace('leftImg8bit.png', 'gtFine_labelIds.png')
                label_save_path = os.path.join(save_label_dir, label_filename)
                self.save_label(lbl, label_save_path)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        logger.info("Data preparation completed.")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Cityscapes Dataset Processing")
    parser.add_argument("--dataset_path", type=str, default="/tmp/ahsan/sqfs/storage_local/datasets/public/cityscapes/", help="Path to the Cityscapes dataset root directory")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train", help="Dataset split to process")
    args = parser.parse_args()

    # Initialize and prepare the dataset
    dataset = cityscapesLoader(root=args.dataset_path, split=args.split)
    dataset.prepare_data()
