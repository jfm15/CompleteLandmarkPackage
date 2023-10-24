import os
from skimage import io
import cv2
import numpy as np
import matplotlib as plt
import numpy as np

class preprocessing_ultrasound():
    def __init__(self):
        self.input_dir = '/data/coml-oxmedis/datasets-in-use/ultrasound-opensource/all_imgs' #'/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/img'
        self.input_txt = None # '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/annotations/txt'

        self.out_pad_dir = '/data/coml-oxmedis/datasets-in-use/ultrasound-opensource/all_imgs_standardsize'#/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/img'
        self.out_pad_txt = None #'/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/crop/txt'

        # self.out_regcrop_dir = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/regcrop/img'
        # self.out_regcrop_txt = '/experiments/datasets-in-use/ultrasound-hip-baby-land-seg/images/regcrop/txt'

    def rotate(self, img, deg): 
        if deg == 90:
            rot_img=np.rot90(img)
        else:
            raise ValueError('Trying to rotate but angle is not 90')
        return rot_img

    def padd_and_crop(self, rotate=False):
        for image in  os.listdir(self.input_dir):
            if image.endswith('.jpg'):
                # read image
                img = cv2.imread(os.path.join(self.input_dir,image))

                if rotate ==False:
                    pass
                else:
                    img = self.rotate(img, rotate)

                #print(os.path.join(input_dir,image))
                old_image_height, old_image_width, channels = img.shape
                print(img.shape)

                # create new image of desired size (largest dimension so they are all on the same track)
                new_image_width = 496
                new_image_height = 600
                color = (0,0,0)
                result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
                #add old photo to array 
                result[0:old_image_height, 0:old_image_width] = img

                #crop array and move txt points
                crop_l_x, crop_u_x = 0 +50, new_image_width - 100
                crop_l_y, crop_u_y = 0 +50, new_image_height - 100
                img_new = result[crop_l_x:crop_u_x,crop_l_y:crop_u_y]
                im_name = image[:-4]

                #open txt file and add lower/upper bound
                if self.input_txt == None:
                    pass
                else:
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
    preprocessing_ultrasound().padd_and_crop(rotate=90)

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




                
                

            
                


        
