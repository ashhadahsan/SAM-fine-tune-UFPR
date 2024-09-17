import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import logging
import argparse
import cv2
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

class cityscapesLoader(torch.utils.data.Dataset):
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
        img_path = self.files[self.split][index].rstrip()

        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)

        lbl = cv2.resize(lbl, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        classes = np.unique(lbl)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

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
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def get_bounding_box(self, mask):
        """Generate bounding box [x_min, y_min, x_max, y_max] for a given mask."""
        indices = np.argwhere(mask)
        if indices.size == 0:
            return None
        y_min, x_min = indices.min(axis=0)
        y_max, x_max = indices.max(axis=0)
        return [float(x_min), float(y_min), float(x_max), float(y_max)]

    def prepare_data(self):
        dataset_annotations = {}
        for idx in tqdm(range(len(self.files[self.split])), desc=f"Processing {self.split} data"):
            img_path = self.files[self.split][idx].rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )
            filename = os.path.basename(img_path)
            try:
                img, lbl = self.__getitem__(idx)
                bboxes = []
                lbl_np = lbl.numpy()
                unique_classes = np.unique(lbl_np)
                for class_id in unique_classes:
                    if class_id == self.ignore_index:
                        continue
                    mask = (lbl_np == class_id).astype(np.uint8)
                    bbox = self.get_bounding_box(mask)
                    if bbox:
                        bboxes.append({
                            "class_id": int(class_id),
                            "bbox": bbox
                        })

                dataset_annotations[filename] = {
                    "image_path": img_path,
                    "mask_path": lbl_path,
                    "bboxes": bboxes
                }
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        logger.info(f"Completed processing {self.split} data.")
        return dataset_annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cityscapes Dataset Processing")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to the dataset root directory", default="/tmp/ahsan/sqfs/storage_local/datasets/public/cityscapes/")
    args = parser.parse_args()

    annotations_path = 'annotations_cityscapes.json'
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {"train": {}, "val": {}}

    for split in ["train", "val"]:
        logging.info(f"Preparing annotations for {split}")
        dataset = cityscapesLoader(root=args.dataset_path, split=split)
        data = dataset.prepare_data()
        annotations[split] = data

    with open(annotations_path, 'w') as f:
        json.dump(annotations, f, indent=4)

    logger.info(f"Annotations saved to {annotations_path}")
