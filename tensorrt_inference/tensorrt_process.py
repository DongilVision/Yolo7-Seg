import time
import cv2
import torch
import numpy as np
from multiprocessing import Queue,Process
from threading import Thread
from engine_torch import BaseEngine
from PIL import Image,ImageDraw,ImageFont

def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)

def check_pil_font():
    return ImageFont.load_default()

def scale_masks(img1_shape, masks, img0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    resize for the most time
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    tl_pad = int(pad[1]), int(pad[0])  # y, x
    br_pad = int(img1_shape[0] - pad[1]), int(img1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    # masks_h, masks_w, n
    masks = masks[tl_pad[0]:br_pad[0], tl_pad[1]:br_pad[1]]
    # 1, n, masks_h, masks_w
    # masks = masks.permute(2, 0, 1).contiguous()[None, :]
    # # shape = [1, n, masks_h, masks_w] after F.interpolate, so take first element
    # masks = F.interpolate(masks, img0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    # masks_h, masks_w, n
    masks = cv2.resize(masks, (img0_shape[1], img0_shape[0]))

    # keepdim
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

def plot_masks(img, masks, colors, alpha=0.5):
    """
    Args:
        img (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
        masks (tensor): predicted masks on cuda, shape: [n, h, w]
        colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
    Return:
        ndarray: img after draw masks, shape: [h, w, 3]

    transform colors and send img_gpu to cpu for the most time.
    """
    img_gpu = img.clone()
    num_masks = len(masks)
    if num_masks == 0:
        return img.permute(1, 2, 0).contiguous().cpu().numpy() * 255

    # [n, 1, 1, 3]
    # faster this way to transform colors
    colors = torch.tensor(colors, device=img.device).float() / 255.0
    colors = colors[:, None, None, :]
    # [n, h, w, 1]
    masks = masks[:, :, :, None]
    masks_color = masks.repeat(1, 1, 1, 3) * colors * alpha
    inv_alph_masks = masks * (-alpha) + 1
    masks_color_summand = masks_color[0]
    if num_masks > 1:
        inv_alph_cumul = inv_alph_masks[:(num_masks - 1)].cumprod(dim=0)
        masks_color_cumul = masks_color[1:] * inv_alph_cumul
        masks_color_summand += masks_color_cumul.sum(dim=0)

    # print(inv_alph_masks.prod(dim=0).shape) # [h, w, 1]
    img_gpu = img_gpu.flip(dims=[0])  # filp channel for opencv
    img_gpu = img_gpu.permute(1, 2, 0).contiguous()
    # [h, w, 3]
    img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    return (img_gpu * 255).byte().cpu().numpy()

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        non_ascii = not is_ascii(example)  # non-latin labels, i.e. asian, arabic, cyrillic
        self.pil = pil or non_ascii
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_pil_font(font='Arial.Unicode.ttf' if non_ascii else font,
                                       size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # box
            if label:
                w, h = self.font.getsize(label)  # text width, height
                outside = box[1] - h >= 0  # label fits outside box
                self.draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=color,
                )
                # self.draw.text((box[0], box[1]), label, fill=txt_color, font=self.font, anchor='ls')  # for PIL>8.0
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)
        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            self.lw / 3,
                            txt_color,
                            thickness=tf,
                            lineType=cv2.LINE_AA)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image (PIL-only)
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def fromarray(self, im):
        # Update self.im from a numpy array
        self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        self.draw = ImageDraw.Draw(self.im)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


class Inference_Unit(Process):
    def __init__(self,input_queue:Queue,output_queue:Queue,engine_path:str,daemon:bool=True):
        super().__init__(daemon=daemon)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.engine_path = engine_path
        self.slice_width = 1280
        self.slice_height = 1280
        self.colors = [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]
        self.start()
    
    def slice_image(self,image):
        p1 = image[:self.slice_height,:self.slice_width,:]
        p2 = image[:self.slice_height,image.shape[1]-self.slice_width:,:]
        p3 = image[image.shape[0]-self.slice_height:,:self.slice_width,:]
        p4 = image[image.shape[0]-self.slice_height:,image.shape[1]-self.slice_width:,:]
        return p1,p2,p3,p4
    
    def expand(self,masks,det,index,w,h):
        new_masks = torch.zeros((len(masks),h,w)).cuda()
        if index==0:
            for i in range(len(masks)):
                new_masks[i,:self.slice_height,:self.slice_width]=masks[i,:,:]
            det[:,0]=det[:,0]+self.slice_width
            det[:,1]=det[:,1]+self.slice_height
            det[:,2]=det[:,2]+self.slice_width
            det[:,3]=det[:,3]+self.slice_height
            return new_masks,det
        elif index==1:
            for i in range(len(masks)):
                new_masks[i,:self.slice_height,w-self.slice_width:]=masks[i,:,:]
            det[:,0]=det[:,0]+(w-self.slice_width)
            det[:,1]=det[:,1]+self.slice_height
            det[:,2]=det[:,2]+(w-self.slice_width)
            det[:,3]=det[:,3]+self.slice_height
            return new_masks,det
        elif index==2:
            for i in range(len(masks)):
                new_masks[i,h-self.slice_height:,:self.slice_width]=masks[i,:,:]
            det[:,0]=det[:,0]+self.slice_width
            det[:,1]=det[:,1]+(h-self.slice_height)
            det[:,2]=det[:,2]+self.slice_width
            det[:,3]=det[:,3]+(h-self.slice_height)
            return new_masks,det
        elif index==3:
            for i in range(len(masks)):
                new_masks[i,h-self.slice_height:,w-self.slice_width:]=masks[i,:,:]
            det[:,0]=det[:,0]+(w-self.slice_width)
            det[:,1]=det[:,1]+(h-self.slice_height)
            det[:,2]=det[:,2]+(w-self.slice_width)
            det[:,3]=det[:,3]+(h-self.slice_height)
            return new_masks,det
        else:
            raise Exception("이상해!!!!!!!!!!!!!!!!!!!!!!!!")
            

    def run(self):
        self.engine = BaseEngine(engine_path=self.engine_path)
        while True:
            try:
                image = self.input_queue.get_nowait()
                annotator = Annotator(image, line_width=3)
                parts = self.slice_image(image)
                results = [self.expand(*self.engine.inference(part,conf=0.3,mask_iou=0.2),index=index,w=image.shape[1],h=image.shape[0]) for index,part in enumerate(parts)]
                masks = torch.concat([ mask for mask,_ in results],dim=0)
                det = torch.concat([ det for _,det in results],dim=0)
                eye = self.engine.nms_mask(masks,det)
                ys,xs = np.where(eye>0.1)
                selected_index = [True]*len(masks)
                for y,x, in zip(ys,xs):
                    if det[y,4].to("cpu").item()>=det[x,4].to("cpu").item():
                        selected_index[x]=False
                    else:
                        selected_index[y]=False
                masks = masks[selected_index]
                det = det[selected_index]
                im_masks = plot_masks((torch.from_numpy(image).permute(2,0,1).cuda().float()/255),masks,[ [0,0,255] for _ in range(masks.shape[0])])
                annotator.im = scale_masks([2048,2448],im_masks,image.shape)
                visualized_image = annotator.result()[:,:,::-1]
                self.output_queue.put(visualized_image)
            except:
                # print("더미중")
                self.engine.dummy_inference()



if __name__=="__main__":
    input_queue = Queue()
    output_queue = Queue()
    image = cv2.imread("bigdog.jpg")
    image = cv2.resize(image,(2448,2048))
    unit = Inference_Unit(input_queue,output_queue,engine_path = "../yolov7-seg-12801280.engine")
    for _ in range(100):
        s = time.time()
        input_queue.put(image)
        visualized_image = output_queue.get()
        e = time.time()
        print(e-s)
        cv2.imwrite("asdf.jpg",visualized_image)
    time.sleep(30)
    
    
    