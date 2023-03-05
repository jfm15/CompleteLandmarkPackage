import os
from skimage import io
import cv2
import numpy as np
import matplotlib as plt

class preprocessing_ultrasound():
    def __init__(self):
        self.input_dir = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/img'
        self.input_txt = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/annotations/txt'

        self.out_pad_dir = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/img'
        self.out_pad_txt = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/txt'

        # self.out_regcrop_dir = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/regcrop/img'
        # self.out_regcrop_txt = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/regcrop/txt'

    def padd_and_crop(self):
        for image in  os.listdir(self.input_dir):
            if image.endswith('.jpg'):
                # read image
                img = cv2.imread(os.path.join(self.input_dir,image))
                #print(os.path.join(input_dir,image))
                old_image_height, old_image_width, channels = img.shape

                # create new image of desired size (largest dimension so they are all on the same track)
                new_image_width = 864
                new_image_height = 608
                color = (0,0,0)
                result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
                #add old photo to array 
                result[0:old_image_height, 0:old_image_width] = img

                #crop array and move txt points
                crop_l_x, crop_u_x = 18, 402
                crop_l_y, crop_u_y = 90, 602
                img_new = result[crop_l_x:crop_u_x,crop_l_y:crop_u_y]
                im_name = image[:-4]

                #open txt file and add lower/upper bound
                orig_txt_= open(self.input_txt+'/'+im_name+'.txt',"r")
                new_txt = self.out_pad_txt+'/'+im_name+".txt"
                if not os.path.isdir(self.out_pad_txt):
                    os.makedirs(self.out_pad_txt)
                with open(new_txt, 'a') as output:
                    orig_txt=orig_txt_.read()
                    orig_txt=orig_txt.split('\n')
                    for i in range(len(orig_txt)-1):
                        row=orig_txt[i].split(',')
                        data_str = str(round(int(row[0])-crop_l_x))+","+str(round(int(row[1])-crop_l_y))
                        data_str = data_str+"\n"
                        output.write(data_str)

                # save 
                #img_new = result
                if not os.path.isdir(self.out_pad_dir):
                    os.makedirs(self.out_pad_dir)
                print(os.path.join(self.out_pad_dir,image))
                cv2.imwrite(os.path.join(self.out_pad_dir,image), img_new)

if __name__ == "__main__":
    preprocessing_ultrasound().padd_and_crop()

# def register(self, im):
#     count = 0
#     for image in  os.listdir(self.input_dir):
#         count += 1
#         _img_tmp = np.array([])

#         if image.endswith('.jpg'):
#             if count == 1:
#                 img_ref = cv2.imread(os.path.join(self.input_dir,image))
#                 _img_tmp.append(img_ref)
#                 h, w = img_ref.shape
#             else:
#                 img_to_align =  cv2.imread(os.path.join(self.input_dir,image))
                
#                 matrix = cv2.getAffineTransform(img_ref,img_to_align)
                
#                 #apply matrix to source image

#             cv2.imshow('ref image', img_ref)
#             cv2.imshow('aligned', img_alignd)




                
                

            
                


        
