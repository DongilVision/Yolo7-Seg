import yaml
import os
from datetime import datetime
# yaml_path = "./config.yaml"
yaml_path = "/data/config.yaml"
with open(yaml_path,"r",encoding="utf-8") as f:
    configs = yaml.load(f,Loader=yaml.FullLoader)
USER_PARAMS = configs["USER_PARAMS"]
DATA_PARAMS = configs["DATA_PARAMS"]
YOLO_PARAMS = configs["YOLO_PARAMS"]
date = datetime.now().strftime("%YY-%mM-%dD-%Hh-%Mmin-%Ssec")
try:
    AUG_PARAMS = configs["AUG_PARAMS"]
    # with open("./config.yaml","w") as f:
    with open("/content/yolov7/config.yaml","w") as f:
        yaml.dump(AUG_PARAMS,f)
    AUG = True
except:
    AUG = False
if USER_PARAMS['WEIGHTS']=="nano":
    weights = "yolov7.pt"
    cfg = "yolov7.yaml"
    batch_size = 12
elif USER_PARAMS['WEIGHTS']=="x":
    weights = "yolov7x.pt"
    cfg = "yolov7x.yaml"
    batch_size = 9
elif USER_PARAMS['WEIGHTS']=="w6":
    weights = ""
    batch_size = 2
    cfg = "yolov7-w6.yaml"
elif USER_PARAMS['WEIGHTS']=="e6":
    weights = ""
    cfg = "yolov7-e6.yaml"
    batch_size = 1
elif USER_PARAMS['WEIGHTS']=="d6":
    weights = ""
    cfg = "yolov7-d6.yaml"
    batch_size = 1
elif USER_PARAMS['WEIGHTS']=="e6e":
    weights = ""
    cfg = "yolov7-e6e.yaml"
    batch_size = 1
train_txt = f"python3 /content/yolov7/segment/train.py\
    --batch-size {batch_size}\
    --epochs {USER_PARAMS['EPOCHS']}\
    --device {USER_PARAMS['DEVICE']}\
    --img-size {USER_PARAMS['IMG-SIZE']}\
    --name {date}\
    --label-smoothing {USER_PARAMS['LABEL-SMOOTHING']}\
    --weights /content/yolov7/{weights}\
    --data '/content/yolov7/data/data.yaml'\
    --project /data/{USER_PARAMS['SAVE-FOLDER-NAME']}\
    --cfg /content/yolov7/cfg/training/{cfg}\
    --mask-ratio 2\
    --hyp '/content/yolov7/data/hyp.yaml'"
export_txt = f"python3 /content/yolov7/export.py \
    --weights /data/USER_PARAMS['SAVE-FOLDER-NAME']/{date}/weights/best.pt\
    --img-size {USER_PARAMS['IMG-SIZE']} {USER_PARAMS['IMG-SIZE']}\
    --simplify\
    --device 0\
    --include onnx\
    --iou-thres 0.25\
    --conf-thres 0.1\
    "
export_trt_path = f"python3 /content/TensorRT-For-YOLO-Series/export.py \
    -o /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights/best.onnx \
    -e /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights/best.trt \
    -p fp16 --conf_thres=0.1 --iou_thres=0.25 --max_det 1000 -w 2"
# with open("./train.sh","w") as f:
with open("/data/train.sh","w") as f:
    f.write("#!/bin/bash\n")
    f.write("\n")
    #f.write("tensorboard --logdir /data/result &")
    f.write("\n")
    if AUG:
        f.write("python3 /content/yolov7/main.py")
        f.write("\n")
    f.write(train_txt)
    f.write("\n")
    f.write(export_txt)
    f.write("\n")
    f.write(export_trt_path)
    f.write("\n")
    f.write(f"mkdir /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights_temp")
    f.write("\n")
    f.write(f"cp /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights/best.pt /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights_temp/best.pt")
    f.write("\n")
    f.write(f"cp /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights/best.onnx /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights_temp/best.onnx")
    f.write("\n")
    f.write(f"cp /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights/best.trt /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights_temp/best.trt")
    f.write("\n")
    f.write(f"rm -rf /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights")
    f.write("\n")
    f.write(f"mv /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights_temp/ /data/{USER_PARAMS['SAVE-FOLDER-NAME']}/{date}/weights")
    f.write("\n")
DATA_PARAMS["train"] = "/data/images/train" if not AUG else "/data/AUG_DATA/images/train"
DATA_PARAMS["val"] = "/data/images/validation" if not AUG else "/data/AUG_DATA/images/validation"
with open("/content/yolov7/data/data.yaml" ,"w") as f:
    yaml.dump(DATA_PARAMS,f)
with open("/content/yolov7/data/hyp.yaml" ,"w") as f:
    yaml.dump(YOLO_PARAMS,f)
