import albumentations as A
import random
import copy
from div_aug_utils import load_config

def albu_augmentations(key,value):
    if key=="Affine":
        return A.Affine(p=value["Probability"],rotate=value["Rotate"],translate_percent=value["Translation"])
    elif key=="Blur":
        object = A.OneOf(
                [
                    A.GlassBlur(p=1,max_delta=2),
                    A.MedianBlur(p=1,blur_limit=7),
                    A.MotionBlur(p=1,blur_limit=7),
                    A.GaussianBlur(p=1,blur_limit=(3,7))
                ],
                p=value["Probability"]
                )
        return object
    elif key=="Brightness":
        return A.RandomBrightnessContrast(p=value["Probability"],brightness_limit=value["Intensity"],contrast_limit=0.0)
    elif key=="Contrast":
        return A.RandomBrightnessContrast(p=value["Probability"],contrast_limit=value["Intensity"],brightness_limit=0.0)
    elif key=="Gamma":
        return A.RandomGamma(p=value["Probability"])
    elif key=="GaussianNoise":
        return A.GaussNoise(p=value["Probability"],var_limit=tuple(value["Intensity"]))
    elif key=="HorizentalFlip":
        return A.HorizontalFlip(p=value["Probability"])
    elif key=="HueTransform":
        return A.HueSaturationValue(p=value["Probability"],hue_shift_limit=value["Hue"],sat_shift_limit=value["Sat"])
    elif key=="RGBShift":
        limit = value["Limit"]
        return A.RGBShift(p=value['Probability'],r_shift_limit=limit,g_shift_limit=limit,b_shift_limit=limit)
    elif key=="PerspectiveTransform":
        return A.Perspective(p=value["Probability"],scale=tuple(value["Scale"]))
    elif key=="RandomCrop":
        return A.RandomCrop(p=value["Probability"],width=value["Crop_size"][0],height=value["Crop_size"][1],)
    elif key=="VerticalFlip":
        return A.VerticalFlip(p=value["Probability"])
        

def config2pipelines():
    config = load_config()
    aug_dict = config["2_AUGMENTATION"]
    pipeline = []
    for key in aug_dict:
        value = aug_dict[key]
        object = albu_augmentations(key,value)
        if object is None:
            pass
        else:
            pipeline.append(object)
    pipelines = []
    for i in range(20):
        random.shuffle(pipeline)
        temp = A.Compose(pipeline,bbox_params=A.BboxParams(format='yolo'))
        pipelines.append(temp)
    return pipelines
        
        
if __name__=="__main__":
    config2pipelines()
    