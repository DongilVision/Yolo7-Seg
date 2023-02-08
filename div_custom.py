import numpy as np
import cv2
import os
import random

def aug_gradation(image,apply=0.5,gradation_images=[]):
    probability = random.random()
    if probability<=float(apply):
        gradation_image = gradation_images[random.randint(0,len(gradation_images)-1)]
        w,h = gradation_image.shape[1]//2,gradation_image.shape[0]//2
        w_random,h_random = random.randint(0,w-1),random.randint(0,h-1)
        temp = gradation_image[h_random:(h_random+h),w_random:(w_random+w),:]
        temp = cv2.resize(temp,dsize=(image.shape[1],image.shape[0]))
        ratio = random.random()*0.5+0.5
        result = cv2.addWeighted(image,ratio,temp,1-ratio,0)
        return result
    else:
        return image
    
def read_gradation_images(config):
    if config["2_AUGMENTATION"]["Gradation"]["Probability"]>0:
        gradation_image_path = "./gradation_images"
        image_list = os.listdir(gradation_image_path)
        image_list = [ cv2.imread(os.path.join(gradation_image_path,name)) for name in image_list]
        return image_list
    else:
        return []
    
def make_gradation_images(config,rgb=(255,255,255)):
    if config["2_AUGMENTATION"]["Gradation"]["Probability"]>0:
        if not os.path.isdir("./gradation_images"):
            os.mkdir("./gradation_images")
            WIDTH = 1280
            HEIGHT = 720
            for i in [1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]:
                dummy_array = np.zeros((2*HEIGHT,2*WIDTH))
                m1,m2 = HEIGHT/WIDTH,-HEIGHT/WIDTH
                for x in range(-WIDTH,WIDTH):
                    for y in range(-HEIGHT,HEIGHT):
                        if y<=m1*x:
                            if y>=m2*x:
                                #1
                                value = (-255/WIDTH)*x+255
                                dummy_array[(y+HEIGHT),(x+WIDTH)]=value
                            else: 
                                #4
                                value = (255/HEIGHT)*y+255
                                dummy_array[(y+HEIGHT),(x+WIDTH)]=value
                        else:
                            if y>=m2*x:
                                value = (-255/HEIGHT)*y+255
                                dummy_array[(y+HEIGHT),(x+WIDTH)]=value
                            else:
                                #3
                                value = (255/WIDTH)*x+255
                                dummy_array[(y+HEIGHT),(x+WIDTH)]=value
                # dummy_array = np.cos(5*np.pi*(dummy_array/255))/2+0.5
                dummy_array = (dummy_array/255)**i
                dummy_array = (dummy_array)*255
                dummy_array = dummy_array.astype(np.uint8)
                cv2.imwrite("./gradation_images/sample_center_gradation_{}.jpg".format(i),dummy_array)

            for j in list(range(3,30,2)):
                dummy_array_cos = np.zeros((2*HEIGHT,2*WIDTH))
                dummy_array_sin = np.zeros((2*HEIGHT,2*WIDTH))
                m1,m2 = HEIGHT/WIDTH,-HEIGHT/WIDTH
                for x in range(-WIDTH,WIDTH):
                    for y in range(-HEIGHT,HEIGHT):
                        if y<=m1*x:
                            if y>=m2*x:
                                #1
                                value = (-255/WIDTH)*x+255
                                dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                                dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
                            else: 
                                #4
                                value = (255/HEIGHT)*y+255
                                dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                                dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
                        else:
                            if y>=m2*x:
                                value = (-255/HEIGHT)*y+255
                                dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                                dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
                            else:
                                #3
                                value = (255/WIDTH)*x+255
                                dummy_array_cos[(y+HEIGHT),(x+WIDTH)]=value
                                dummy_array_sin[(y+HEIGHT),(x+WIDTH)]=value
                dummy_array_cos = np.cos(j*np.pi*(dummy_array_cos/255))/2+0.5
                dummy_array_other = np.sin(np.sin(np.sin(np.sin(np.sin(j*np.pi*(dummy_array_sin/255))/2+0.5))))
                dummy_array_cos = (dummy_array_cos)*255
                dummy_array_cos = dummy_array_cos.astype(np.uint8)
                dummy_array_other = (dummy_array_other)*255
                dummy_array_other = dummy_array_other.astype(np.uint8)
                
                cv2.imwrite("./gradation_images/sample_center_gradation_cos{}.jpg".format(j),dummy_array_cos)
                cv2.imwrite("./gradation_images/sample_center_gradation_other{}.jpg".format(j),dummy_array_other)
        else:
            pass