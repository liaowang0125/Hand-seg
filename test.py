import argparse

import numpy
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable

from unet import UNet
from utils import *


def predict_img(net, full_img, gpu=False):
    # img = resize_and_crop(full_img)
    # # print(img.size,full_img.size)
    # left = get_square(img, 0)
    # right = get_square(img, 1)

    # right = normalize(right)
    # left = normalize(left)

    # right = np.transpose(right, axes=[2, 0, 1])
    # left = np.transpose(left, axes=[2, 0, 1])

    # X_l = torch.FloatTensor(left).unsqueeze(0)
    # X_r = torch.FloatTensor(right).unsqueeze(0)

    # if gpu:
    #     X_l = Variable(X_l, volatile=True).cuda()
    #     X_r = Variable(X_r, volatile=True).cuda()
    # else:
    #     X_l = Variable(X_l, volatile=True)
    #     X_r = Variable(X_r, volatile=True)

    # y_l = F.sigmoid(net(X_l))
    # y_r = F.sigmoid(net(X_r))
   
    # y_l = F.upsample_bilinear(y_l, scale_factor=2).data[0][0].cpu().numpy()
    # y_r = F.upsample_bilinear(y_r, scale_factor=2).data[0][0].cpu().numpy()

    # y = merge_masks(y_l, y_r, full_img.size[0])
    # yy = dense_crf(np.array(full_img).astype(np.uint8), y)

    # return yy > 0.5

    full_img= full_img.resize((224, 224))
    img=transform_test(full_img)
    # img=img.unsqueeze_(0)
    # img = resize_and_crop(full_img)
    # img=np.array(full_img)
    # img = normalize(img)
    # img = np.transpose(img, axes=[2, 0, 1])
    # img=full_img
    # img = np.transpose(img, axes=[2, 0, 1])
    img = torch.FloatTensor(img).unsqueeze(0)

    if gpu:
        img = Variable(img, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)

    img = F.sigmoid(net(img))

    img = F.upsample_bilinear(img, scale_factor=1).data[0][0].cpu().numpy()
    # print(full_img.size,img.shape)
    yy = dense_crf(np.array(full_img).astype(np.uint8), img)

    return yy > 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_false',
                        help="Do not save the output masks",
                        default=False)

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 1)
    net.eval()
    if not args.cpu:
        torch.cuda.set_device(1)
        print("Using CUDA version of the net, prepare your GPU !")

        net.cuda()
    else:
        net.cpu()
        print("Using CPU version of the net, this may be very slow")

    in_files = args.input
    print("in_files=",in_files)
    out_files = []
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))
    print("Model loaded !")

    transform_test = transforms.Compose([
            transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        img = Image.open(fn).convert('RGB')
        
        out = predict_img(net, img, not args.cpu)
        if args.viz:
            print("Vizualising results for image {}, close to continue ..."
                  .format(fn))
            plot_img_mask(img, out)
        if not args.no_save:
            out_fn = out_files[i]
            result = Image.fromarray((out * 255).astype(numpy.uint8))
            result.save(out_files[i])
              
            print("Mask saved to {}".format(out_files[i]))
