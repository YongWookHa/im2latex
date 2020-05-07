import numpy as np
from torchvision import transforms
from PIL import Image
from datetime import datetime

def fetch_cosine_values(seq_len, frequency=0.01, noise=0.1):
    np.random.seed(101)
    x = np.arange(0.0, seq_len, 1.0)
    return np.cos(2 * np.pi * frequency * x) + np.random.uniform(low=noise, high=noise, size=seq_len)

def format_dataset(values, temporal_features):
    feat_splits = [values[i:i+temporal_features] for i in range(len(values) - temporal_features)]
    feats = np.array(feat_splits)
    labels = np.array(values[temporal_features:])
    return feats, labels

def matrix_to_array(m):
    return np.asarray(m).reshape(-1)

def read_classes(fn):
    with open(fn, 'r', encoding='utf8') as f:
        classes = []
        line = f.readline().strip()
        while line:
            classes.append(line)
            line = f.readline().strip()
    assert len(classes) != 0
    print("number of detected classes : {}".format(len(classes)))
    return classes

def read_image(fn, size:tuple):
    # this function reads an image
    img = Image.open(fn)
    transformations = transforms.Compose([
                            transforms.Resize(size, Image.BICUBIC),
                            transforms.ToTensor()
                            ])
    return transformations(img).unsqueeze(0)                     
    