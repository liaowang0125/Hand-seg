#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
from functools import partial
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from .utils import resize_and_crop, get_square, normalize


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    a=[]
    for f in os.listdir(dir):
        a.append(f[:-4])
    return a
    # return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    # for id, pos in ids:
    #     im = resize_and_crop(Image.open(dir + id + suffix))
    #     yield get_square(im, pos)

    for id in ids:
        im = np.array(Image.open(dir + id + suffix))
        yield im
def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    # imgs = to_cropped_imgs(ids, dir_img, '.jpg')

    # # need to transform from HWC to CHW
    # imgs_switched = map(partial(np.transpose, axes=[2, 0, 1]), imgs)
    # imgs_normalized = map(normalize, imgs_switched)

    # masks = to_cropped_imgs(ids, dir_mask, '.png')

    # return zip(imgs_normalized, masks)

    
    # transform_train_list = [
    #         transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0),
    #         transforms.ToTensor()   
    #         ]
    # data_transforms=transforms.Compose( transform_train_list )
    data=[]
    for id in ids:
        img=Image.open(dir_img + id + '.jpg').convert('RGB')
        # img=data_transforms(img)
        # img=np.array(img)
        label=np.array(Image.open(dir_mask + id + '.png'))
        data.append((img,label))
    return data

def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '.png')
    return np.array(im), np.array(mask)

class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img, pid = self.dataset[item]
        # img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid

    def __len__(self):
        return len(self.dataset)