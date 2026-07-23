import os
import csv
import numpy as np
import jax.numpy as jnp
from PIL import Image
import random
import categories

DATA_DIR = "butterfly-data"


def transform(img, mean, std):
    # Resize if needed
    if img.size != (224, 224):
        img = img.resize((224, 224))

    # Convert to numpy float32
    img = np.array(img).astype(np.float32) / 255.0

    # Normalize
    img = (img - mean) / std

    return img

class ImageDataset:

    def __init__(self, is_train=True):
        # Map category string to integer
        self.cat_dict = {category: i for i, category in enumerate(categories.categories)}

        if is_train:
            csvname = "Training_set.csv"
            prefix = "train"
            self.is_train = True
        else:
            csvname = "Validation_set.csv"
            prefix = "validate"
            self.is_train = False

        # Read CSV
        path = os.path.join(DATA_DIR, csvname)
        image_list = []
        label_list = []

        with open(path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for row in reader:
                filename = row[0].strip()
                category = row[1].strip()
                image_list.append(os.path.join(DATA_DIR, prefix, filename))
                label_list.append(self.cat_dict[category])

        self.x = image_list
        self.y = label_list

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.x)

    def _transform(self, img):
        # Random horizontal flip (train only)
        if self.is_train and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return transform(img, self.mean, self.std)

    def __getitem__(self, index):
        filename = self.x[index]
        img = Image.open(filename).convert("RGB")

        img = self._transform(img)

        label = self.y[index]

        # Return JAX arrays
        return jnp.array(img), jnp.array(label)