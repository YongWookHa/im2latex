from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

import random
import numpy as np
import cv2
import pickle
import torch


class ImageProcessor3(object):
    def __init__(self, params, image_dir_, grayscale):
        self._params=params
        self._image_dir = image_dir_
        self._mode = 'L' if grayscale else 'RGB'
        self._channels = 1 if grayscale else 3

    def get_array(self, image_name_, height_, width_, padded_dim_):
        image_file = Path(self._image_dir, image_name_)
        padded_height = padded_dim_['height']
        padded_width = padded_dim_['width']
        ## Load image and convert to a n-channel array
        im_ar = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if len(im_ar.shape) == 2:
            im_ar = np.expand_dims(im_ar, 2)

        height, width, channels = im_ar.shape
        assert height == height_, 'image height = %d instead of %d'%(height_, height)
        assert width == width_, 'image width = %d instead of %d'%(width_, width)
        assert channels == self._channels, 'image channels = %d instead of %d'%(self._channels, channels)
        assert height < padded_height and width < padded_width, \
            'Image is too large, shape = {shape}, max={max}'.format(shape=im_ar.shape, max=padded_dim_)
        if (height < padded_height) or (width < padded_width):
            ar = np.full((padded_height, padded_width, channels), 255, dtype=self._params['dtype_np'])
            h = (padded_height - height) // 2
            w = (padded_width - width) // 2
            ar[h:h+height, w:w+width] = im_ar
            im_ar = ar

        return im_ar

    def whiten(self, image_ar):
        """
        normalize values to lie between -1.0 and 1.0.
        This is done in place of data whitening - i.e. normalizing to mean=0 and std-dev=0.5
        Is is a very rough technique but legit for images. We assume that the mean is 255/2
        and therefore substract 127.5 from all values. Then we divid everything by 255 to ensure
        that all values lie between -0.5 and 0.5
        Arguments:
            image_batch: (ndarray) Batch of images or a single image. Shape doesn't matter.
        """
        return (image_ar - 127.5) / 255.0

if __name__ == "__main__":
    params = {
                'dtype_np' : np.int16,
                'batch_size' : None,
                'image_shape' : None,
                'train_test_ratio' : 0.95
            }
    image_dir = Path('../data/dataset5/formula_images')
    ip = ImageProcessor3(params, image_dir, grayscale=True)
    transform = transforms.ToTensor()
    total = []

    data_props = pickle.load(open('../data/dataset5/training_56/data_props.pkl', 'rb'), encoding='latin1')

    for split in ['train', 'valid', 'test']:
        data = pickle.load(open('../data/dataset5/training_56/df_{}.pkl'.format(split), 'rb'), encoding='latin1')
        total.extend(list(zip(data['image'], data['height'], data['width'], data['latex_ascii'])))


    random.shuffle(total)

    num_train = int(len(total)*params['train_test_ratio'])
    train = total[:num_train]
    valid = total[num_train:]

    # print('dumping train data with pickle')
    # torch.save(valid, 'valid0507.pkl')

    # print('dumping valid data with pickle')
    # torch.save(train, 'train0507.pkl')
    Path('../data/dataset5/processed/train_imgs').mkdir(parents=True)
    Path('../data/dataset5/processed/valid_imgs').mkdir(parents=True)
    train_file = open('../data/dataset5/processed/train.txt', 'w', encoding='utf8')
    for i, tup in tqdm(enumerate(train), total=len(train), desc='save train'):
        im_fn, h, w, l = tup
        img = ip.get_array(im_fn, h, w, padded_dim_={'height': 128, 'width': 1088})
        cv2.imwrite('../data/dataset5/processed/train_imgs/{}.png'.format(i), img)
        train_file.write(l+'\n')
    train_file.close()

    valid_file = open('../data/dataset5/processed/valid.txt', 'w', encoding='utf8')
    for i, tup in tqdm(enumerate(valid), total=len(valid), desc='save valid'):
        im_fn, h, w, l = tup
        img = ip.get_array(im_fn, h, w, padded_dim_={'height': 128, 'width': 1088})
        cv2.imwrite('../data/dataset5/processed/valid_imgs/{}.png'.format(i), img)
        valid_file.write(l+'\n')
    valid_file.close()

