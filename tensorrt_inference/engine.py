import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time
import numba

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    def crop(self,masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def sigmoid(self,array):
        return 1/(1+np.exp(-array))

    def process_mask(self,protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        protos_copy = protos.copy()
        protos_copy = protos_copy.reshape(c,-1)
        masks = self.sigmoid(masks_in @ protos_copy).reshape(-1, mh, mw)  # CHW
        downsampled_bboxes = bboxes.copy()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih
        masks = self.crop(masks, downsampled_bboxes)  # CHW
        
        if upsample and (len(masks)>0):
            masks = cv2.resize(np.transpose(masks,(1,2,0)),shape)
        return np.where(masks>0.5,True,False)
    
    def non_max_suppression(self,prediction,conf_thres=0.25,iou_thres=0.45,classes=None,agnostic=False,multi_label=False,labels=(),max_det=300,nm=0,):
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        
        t = time.time()
        mi = 5 + nc  # mask start index
        output = [np.zeros((0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]] 
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = np.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), 0)

            if not x.shape[0]:
                continue

            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = np.concatenate((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(keepdims=True,axis=1),x[:, 5:mi].argmax(keepdims=True,axis=1)
                x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > conf_thres]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort()[::-1]]  # sort by confidence
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = np.array(range(len(x)))
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output
    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data
    
    def inference(self, origin_img, conf=0.25, iou=0.2):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        a = time.time()
        origin_h,origin_w,origin_c = origin_img.shape
        img = self.preproc(origin_img, self.mean, self.std)
        b = time.time()
        print("전처리 : ", b-a)
        data = self.infer(img)
        c = time.time()
        print("추론",c-b)
        proto,_,_,_,pred = data
        pred = pred.reshape(1,-1,117)
        proto = proto.reshape(1,32,self.imgsz[0]//4,self.imgsz[1]//4)
        pred = self.non_max_suppression(pred,conf_thres=conf,iou_thres=iou,nm=32)
        d = time.time()
        print("후처리 1 nms: ",d-c)
        det = pred[0]
        masks = self.process_mask(proto[0],det[:,6:],det[:,:4],self.imgsz[::-1],upsample=True)
        e = time.time()
        print("후처리 2 nms: ",e-d)
        if len(masks)<1:
            return dict()
        eye = self.masks_nms(masks,det)
        ys,xs = np.where(eye>iou)
        selected_index = [True]*masks.shape[-1]
        for y,x, in zip(ys,xs):
            if det[y,4]>=det[x,4]:
                selected_index[x]=False
            else:
                selected_index[y]=False
        masks = cv2.resize(masks[:,:,selected_index].astype(np.uint8)*255,(origin_w,origin_h))
        det = det[selected_index]
        d = time.time()
        print("후처리 : ",d-c)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        return dict()

    def masks_nms(self,masks,det):
        if len(det)==1:
            return np.eye(len(det))
        else:
            eye = np.eye(len(det))
            for index in range(len(det)):
                mask = masks[:,:,index]
                after_masks = masks[:,:,index+1:]
                sum_maps = np.add(mask[:,:,None],after_masks)
                mul_maps = np.multiply(mask[:,:,None],after_masks)
                sums = np.sum(np.sum(sum_maps,axis=0),axis=0)
                muls = np.sum(np.sum(mul_maps,axis=0),axis=0)
                eye[index,index+1:]=muls/sums
            return eye - np.eye(len(det))
    
    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')
    
    def preproc(self,image, mean, std, swap=(2, 0, 1)):
        # if len(image.shape) == 3:
        #     padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        # else:
        #     padded_img = np.ones(input_size) * 114.0
        # img = np.array(image)
        # r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        # resized_img = cv2.resize(
        #     img,
        #     (int(img.shape[1] * r), int(img.shape[0] * r)),
        #     interpolation=cv2.INTER_LINEAR,
        # ).astype(np.float32)
        # h = int(img.shape[0] * r)
        # w = int(img.shape[1] * r)
        # padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = cv2.resize(image,self.imgsz[::-1]).astype(np.float32)
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

if __name__=="__main__":
    engine = BaseEngine("../yolov7-seg-fp16.engine")
    origin_img = "dog.png"
    image = cv2.imread(origin_img)
    while True:
        engine.inference(image)