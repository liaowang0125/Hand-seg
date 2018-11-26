from PIL import Image
import os
import numpy as np

def change(dir):
    file_list = os.listdir(dir)
    # print(file_list)
    for filename in file_list:
        path = dir+filename
        img = np.array(Image.open(path).convert("L"))
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if(img[i,j]>20):
                    img[i,j]=1
                else:
                    img[i,j]=0
        out = Image.fromarray(img)
                        
        print("%s has been recolored!"%filename)
        out.save(path)

if __name__ == '__main__':
#   dir = input('please input the operate dir:')
   dir = "/home/liaowang/Hand-Segmentation/train_images/masks_1/"
   change(dir)
