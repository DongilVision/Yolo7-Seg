import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision
import time

class BaseEngine(object):
    def __init__(self, engine_path,device="cuda"):
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
        self.device = device
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
        self.dummy_img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        self.dummy_img = np.ascontiguousarray(self.dummy_img, dtype=np.float32)

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

    def postproc(self,data,conf,box_iou,mask_iou,origin_img_shape,polypoint_nums):
        proto,_,_,_,pred = data
        pred = pred.reshape(1,-1,117)
        proto = proto.reshape(1,32,self.imgsz[0]//4,self.imgsz[1]//4)
        proto = torch.Tensor(proto).to(self.device)
        pred = torch.Tensor(pred).to(self.device)
        pred = self.non_max_suppression(pred, conf, box_iou, None, False, max_det=100, nm=32)
        i = 0
        det = pred[0]
        masks = self.process_mask(proto[i], det[:, 6:], det[:, :4], self.imgsz, upsample=True)  # HWC
        if masks=="NONE":
            return torch.zeros((0,*self.imgsz)),torch.zeros((0,38)).cuda()
        det[:, :4] = self.scale_coords(self.imgsz, det[:, :4], origin_img_shape).round()
        return masks,det
        # eye = self.nms_mask(masks,det)
        # ys,xs = np.where(eye>mask_iou)
        # selected_index = [True]*len(masks)
        # for y,x, in zip(ys,xs):
        #     if det[y,4].to("cpu").item()>=det[x,4].to("cpu").item():
        #         selected_index[x]=False
        #     else:
        #         selected_index[y]=False
        # masks = masks[selected_index]
        # det = det[selected_index] # det x,y,x,y,confidence,cls,
        # det[:, :4] = self.scale_coords(self.imgsz, det[:, :4], origin_img_shape).round()
        # polygons = [ cv2.findContours((mask*255).cpu().numpy().astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0] for mask in masks]
        # polygons = [ polygon[-1].reshape(-1,2)[::max(polygon[-1].shape[0]//(polypoint_nums-1),1)] for polygon in polygons]
        return masks
    
    def inference(self, origin_img, conf=0.01, box_iou=0.2,mask_iou=0.2,polypoint_nums=32):
        origin_img_shape = origin_img.shape
        img = self.preproc(origin_img, self.mean, self.std)
        data = self.infer(img)
        return self.postproc(data,conf,box_iou,mask_iou,origin_img_shape,polypoint_nums)
        
    # def inference(self, origin_img, conf=0.01, box_iou=0.2,mask_iou=0.2,polypoint_nums=32):
    #     origin_img_shape = origin_img.shape
    #     img = self.preproc(origin_img, self.mean, self.std)
    #     data = self.infer(img)
    #     proto,_,_,_,pred = data
    #     pred = pred.reshape(1,-1,117)
    #     proto = proto.reshape(1,32,self.imgsz[0]//4,self.imgsz[1]//4)
    #     proto = torch.Tensor(proto).to(self.device)
    #     pred = torch.Tensor(pred).to(self.device)
    #     pred = self.non_max_suppression(pred, conf, box_iou, None, False, max_det=100, nm=32)
    #     i = 0
    #     det = pred[0]
    #     masks = self.process_mask(proto[i], det[:, 6:], det[:, :4], self.imgsz, upsample=True)  # HWC
    #     if masks=="NONE":
    #         return []
    #     eye = self.nms_mask(masks,det)
    #     ys,xs = np.where(eye>mask_iou)
    #     selected_index = [True]*len(masks)
    #     for y,x, in zip(ys,xs):
    #         if det[y,4].to("cpu").item()>=det[x,4].to("cpu").item():
    #             selected_index[x]=False
    #         else:
    #             selected_index[y]=False
    #     masks = masks[selected_index]
    #     det = det[selected_index] # det x,y,x,y,confidence,cls,
    #     det[:, :4] = self.scale_coords(self.imgsz, det[:, :4], origin_img.shape).round()
    #     polygons = [ cv2.findContours((mask*255).cpu().numpy().astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0] for mask in masks]
    #     polygons = [ polygon[-1].reshape(-1,2)[::max(polygon[-1].shape[0]//(polypoint_nums-1),1)] for polygon in polygons]
    #     return polygons
    
    def nms_mask(self,masks,det=None):
        if len(masks)==1:
            return torch.eye(len(masks)).to('cpu').numpy()
        else:
            eye = torch.eye(len(masks)).to(self.device)
            for index,mask_ in enumerate(masks[:-1]):
                mask = mask_.bool()
                after_masks = masks[index+1:,:,:].bool()
                sum_maps = mask+after_masks
                mul_maps = mask*after_masks
                sums = torch.sum(torch.sum(sum_maps,axis=1),axis=1)
                muls = torch.sum(torch.sum(mul_maps,axis=1),axis=1)
                eye[index,index+1:]=muls/sums
            return eye.to("cpu").numpy() - np.eye(len(masks))
    
    def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords    
    
    def clip_coords(self,boxes, shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[:, 0].clamp_(0, shape[1])  # x1
            boxes[:, 1].clamp_(0, shape[0])  # y1
            boxes[:, 2].clamp_(0, shape[1])  # x2
            boxes[:, 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    
    def crop(self,masks, boxes):
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    
    def process_mask(self,protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape  # CHW
        ih, iw = shape
        masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW)
        downsampled_bboxes = bboxes.clone()
        downsampled_bboxes[:, 0] *= mw / iw
        downsampled_bboxes[:, 2] *= mw / iw
        downsampled_bboxes[:, 3] *= mh / ih
        downsampled_bboxes[:, 1] *= mh / ih
        masks = self.crop(masks, downsampled_bboxes)  # CHW
        if upsample and len(masks)>0:
            masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        else:
            return "NONE"
        return masks.gt_(0.5)
    
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
    
    def dummy_inference(self):
        self.infer(self.dummy_img)
        
    def preproc(self,image, mean, std, swap=(2, 0, 1)):
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
    
    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def non_max_suppression(
            self,
            prediction,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            multi_label=False,
            labels=(),
            max_det=300,
            nm=0,  # number of masks
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            else:
                x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output

if __name__=="__main__":
    engine = BaseEngine("../yolov7-seg-12801280.engine")
    engine.get_fps()
    image = cv2.imread("dog.png")
    while True:
        engine.inference(image)