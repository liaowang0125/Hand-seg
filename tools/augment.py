import numpy as np
import cv2
import math
import random
#from meghair.utils.misc import ic01_to_i01c, i01c_to_ic01, list2nparray

class Config:
    resize_shape=(70,105)  #resize (w,h)
    image_shape = (96, 48)
config=Config()

def augment(img,config, do_training,rng=np.random):
    # img = imgproc.resize_preserve_aspect_ratio(img, config.image_shape)

    if do_training:
        # data augmentation from fb.resnet.torch
        # https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua


        def random_crop_im(im,new_size,prng = np.random):
            if(new_size[1] ==im.shape[1]) and (new_size[0] == im.shape[0]):
                return im
            im=cv2.resize(im,config.resize_shape,interpolation=cv2.INTER_CUBIC)
            # print(im.shape,new_size)
            h_start = prng.randint(0, im.shape[0] - new_size[0])
            w_start = prng.randint(0, im.shape[1] - new_size[1])
            im = im[h_start: h_start + new_size[0], w_start: w_start + new_size[1]]
            # im = im[im.shape[0]-new_size[0]-h_start: im.shape[0]-h_start, w_start: w_start + new_size[1]]

            return im

        def scale(img, size):
            s = size / min(img.shape[0], img.shape[1])
            h, w = int(round(img.shape[0] * s)), int(round(img.shape[1] * s))
            return cv2.resize(img, (w, h))

        def center_crop(img, shape):
            h, w = img.shape[:2]
            sx, sy = (w - shape[1]) // 2, (h - shape[0]) // 2
            img = img[sy:sy + shape[0], sx:sx + shape[1]]
            return img

        def random_sized_crop(img):
            NR_REPEAT = 1

            h, w = img.shape[:2]
            area = h * w
            ar = [7. / 8, 8. / 7]
            for i in range(NR_REPEAT):
                target_area = rng.uniform(0.5, 1.0) * area
                target_ar = rng.choice(ar)
                nw = int(round((target_area * target_ar) ** 0.5))
                nh = int(round((target_area / target_ar) ** 0.5))

                if rng.rand() < 0.5:
                    nh, nw = nw, nh

                if nh <= h and nw <= w:
                    sx, sy = rng.randint(w - nw + 1), rng.randint(h - nh + 1)
                    img = img[sy:sy + nh, sx:sx + nw]
                    return cv2.resize(img, config.image_shape[::-1])

            size = min(config.image_shape[0], config.image_shape[1])
            return center_crop(scale(img, size), config.image_shape)

        def grayscale(img):
            w = np.array([0.114, 0.587, 0.299]).reshape(1, 1, 3)
            gs = np.zeros(img.shape[:2])
            gs = (img * w).sum(axis=2, keepdims=True)

            return gs

        def brightness_aug(img, val):
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha

            return img

        def contrast_aug(img, val):
            gs = grayscale(img)
            gs[:] = gs.mean()
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img

        def saturation_aug(img, val):
            gs = grayscale(img)
            alpha = 1. + val * (rng.rand() * 2 - 1)
            img = img * alpha + gs * (1 - alpha)

            return img

        def color_jitter(img, brightness, contrast, saturation):
            augs = [(brightness_aug, brightness),
                    (contrast_aug, contrast),
                    (saturation_aug, saturation)]
            rng.shuffle(augs)

            for aug, val in augs:
                img = aug(img, val)

            return img

        def lighting(img, std):
            eigval = np.array([0.2175, 0.0188, 0.0045])
            eigvec = np.array([
                [-0.5836, -0.6948,  0.4203],
                [-0.5808, -0.0045, -0.8140],
                [-0.5675, 0.7192, 0.4009],
            ])
            if std == 0:
                return img

            alpha = rng.randn(3) * std
            bgr = eigvec * alpha.reshape(1, 3) * eigval.reshape(1, 3)
            bgr = bgr.sum(axis=1).reshape(1, 1, 3)
            img = img + bgr

            return img

        def horizontal_flip(img, prob):
            if rng.rand() < prob:
                return img[:, ::-1]
            return img

        def warp_perspective(img):
            c = (
                ((-50, 50), (-10, 10)),
                ((-50, 50), (-10, 10)),
                ((-50, 50), (-10, 10)),
                ((-50, 50), (-10, 10))
            )
            mat = imgaug.get_random_perspective_transform_mat(
                rng, c, config.image_shape)
            return cv2.warpPerspective(img, mat, config.image_shape)
        def RandomErasing(img,probability=0.5, sl=0.02, sh=0.05, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
            if random.uniform(0, 1) > probability:
                return img
            # print (img.shape)
            for attempt in range(50):
                area = img.shape[1] * img.shape[0]

                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1 / r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)
                    if img.shape[2] == 3:
                        img[x1:x1 + h, y1:y1 + w,0] = mean[0]
                        img[x1:x1 + h, y1:y1 + w,1] = mean[1]
                        img[x1:x1 + h, y1:y1 + w,2] = mean[2]
                    else:
                        img[x1:x1 + h, y1:y1 + w,0] = mean[0]
                    return img

            return img

        # img = color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2)
        # img = random_sized_crop(img)
        # img = lighting(img, 0.7)
        # img = brightness_aug(img, 0.1)
        # img = random_crop_im(img, config.image_shape)
        # img = horizontal_flip(img, 0.5)
        # img=RandomErasing(img,probability=1)
        # img = warp_perspective(img)
        img=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        img = np.minimum(255, np.maximum(0, img))

    #return np.rollaxis(img, 2).astype('float32')
    return img