# Adapted from dataset loader written by meetshah1995 with modifications
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
ROOT_PATH="/tmp/ahsan/sqfs/storage_local/datasets/public/cityscapes/"
path_data = ROOT_PATH
learning_rate = 1e-6
train_epochs = 8
n_classes = 19
batch_size = 1
num_workers = 1
import torch
import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class cityscapesLoader(data.Dataset):
    colors = [  # [  0,   0,   0],
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

    # makes a dictionary with key:value. For example 0:[128, 64, 128]
    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        root,
        # which data split to use
        split="train",
        # transform function activation
        is_transform=True,
        # image_size to use in transform function
        img_size=(512, 1024),
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}

        # makes it: /raid11/cityscapes/ + leftImg8bit + train (as we named the split folder this)
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)
        
        # contains list of all pngs inside all different folders. Recursively iterates 
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        
        # these are 19
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
        ]
        
        # these are 19 + 1; "unlabelled" is extra
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
        
        # for void_classes; useful for loss function
        self.ignore_index = 250
        
        # dictionary of valid classes 7:0, 8:1, 11:2
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        
        # prints number of images found
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
    # path of image
        img_path = self.files[self.split][index].rstrip()

        # path of label
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        # read image using OpenCV
        img = cv2.imread(img_path)
        # convert BGR to RGB (as OpenCV loads images in BGR format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # read label using OpenCV
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        # encode using encode_segmap function: 0...18 and 250
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
    # Image resize using OpenCV
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)  # Resize image

        # No need to convert from BGR to RGB here because we did that in the __getitem__ method
        img = img.astype(np.float64)  # Convert to float64 for further processing
        img = img.transpose(2, 0, 1)  # NHWC -> NCHW

        # Resize label using OpenCV (use nearest neighbor for label resizing)
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

    # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    def encode_segmap(self, mask):
        # !! Comment in code had wrong informtion
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
train_data = cityscapesLoader(
    root = path_data, 
    split='train'
    )
val_data = cityscapesLoader(
    root = path_data, 
    split='val'
    )
train_loader = DataLoader(
    train_data,
    batch_size = batch_size,
    shuffle=True,
    num_workers = num_workers,
)
val_loader = DataLoader(
    val_data,
    batch_size = batch_size,
    num_workers = num_workers,
)

# Function to visualize a single image and segmentation map
def visualize_sample(image, label, loader):
    # Move image and label to CPU (in case they are on GPU)
    image = image.cpu().numpy()
    label = label.cpu().numpy()

    # Transpose image from (C, H, W) to (H, W, C) for displaying
    image = image.transpose(1, 2, 0)

    # Plot image
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image.astype(np.uint8))  # convert to uint8 for display
    plt.title("Input Image")

    # Decode segmentation map
    decoded_label = loader.dataset.decode_segmap(label)

    # Plot the decoded segmentation map
    plt.subplot(1, 2, 2)
    plt.imshow(decoded_label)
    plt.title("Segmentation Label")
    plt.show()

# Get a sample batch from the dataloader
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Select the first image and label from the batch
image = images[0]
label = labels[0]

# Visualize the sample
visualize_sample(image, label, train_loader)
