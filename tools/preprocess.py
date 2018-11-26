from augment import config
from augment import augment
import cv2
import os

origin_data1='Masks_255'
# origin_data0='/home/liaowang/hands_classify/data2__/0'
preprocess_data1='Masks_1_224'
# preprocess_data0='/home/liaowang/hands_classify/data2__/pre0'

if not os.path.exists(preprocess_data1):
	os.mkdir(preprocess_data1)
# if not os.path.exists(preprocess_data0):
# 	os.mkdir(preprocess_data0)
	
file1=os.listdir(origin_data1)
# file0=os.listdir(origin_data0)

for i,name in enumerate(file1):
	print('processing1 %d'%(i))
	origin_path=origin_data1+'/'+name
	# dst_path_ori=preprocess_data1+'/ori1_'+name
	dst_path_pre=preprocess_data1+'/'+name
	ori_img=cv2.imread(origin_path)
	dst_img=augment(ori_img,config,True)
	# cv2.imwrite(dst_path_ori,ori_img)
	cv2.imwrite(dst_path_pre,dst_img)

# for i,name in enumerate(file0):
# 	print('processing0 %d'%(i))
# 	origin_path=origin_data0+'/'+name
# 	dst_path_ori=preprocess_data0+'/ori0_'+name
# 	dst_path_pre=preprocess_data0+'/pre0_'+name
# 	ori_img=cv2.imread(origin_path)
# 	dst_img=augment(ori_img,config,True)
# 	cv2.imwrite(dst_path_ori,ori_img)
# 	cv2.imwrite(dst_path_pre,dst_img)

# img=cv2.imread('/home/liaowang/hands_classify/data2__/1/1__1691.jpg')
# img_process=augment(img,config,True)
# cv2.imwrite('./origin.jpg',img)
# cv2.imwrite('./process.jpg',img_process)