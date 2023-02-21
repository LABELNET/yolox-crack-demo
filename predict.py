"""
推理与后处理 
---
测试模型下载地址
---
https://drive.google.com/file/d/1I4JDTDgiU_ZnSNxH8z9M6pRtC_4cA3Af/view?usp=sharing
"""

import cv2
import cv2

import torch
import os
import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.utils import  postprocess


COCO_CLASSES = ["0"]
CKPT_FILE = 'models/best_ckpt.pth'
# 测试图片，修改此路径，进行测试
TEST_IMAGE = 'asserts/0017.jpg'


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img.copy()

        ratio = min(self.test_size[0] / img.shape[0],
                    self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = self.vis_contours(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

    def vis_contours(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        """
        根据 Box 抽取裂纹，并进行标记处理
        """

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            # 裂纹标记：区域截取，抽取轮廓后，重新合成
            box_img = img[y0:y1,x0:x1]
            # if len(box_img) >0:
            #     cv2.imwrite(f're/re_{i}.png',box_img)
            box_img_contours = image_contours(box_img)
            img[y0:y1,x0:x1] = box_img_contours
            # 标记框
            color = (0,255,0)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        return img




def predict():
    """
    推理预测 
    """
    from crack_exp_yolox_s import Exp
    exp = Exp()
    exp.test_conf = 0.25
    exp.nmsthre = 0.35
    exp.test_size = (640,640)
    # 模型信息
    model = exp.get_model()
    # print(get_model_info(model, exp.test_size))
    model.eval()
    ckpt = torch.load(CKPT_FILE, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print("loaded checkpoint done.")
    # 设备信息
    device = 'cpu'
    fp16 = True
    trt_file = None
    decoder = None
    legacy = False
    # 推理方法
    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        device, fp16, legacy,
    )
    image = cv2.imread(TEST_IMAGE)
    outputs, img_info = predictor.inference(image)
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    cv2.imwrite('result.jpg',result_image)



def image_contours(image):
    """ 
    图片区域标记轮廓，RGB 阈值替换
    """
    color = [0,255,0]
    r_img,g_img,b_img = image[:,:,0].copy(),image[:,:,1].copy(),image[:,:,2].copy()
    # 169 为阈值，具体根据需求自行调整
    r_img[r_img < 169] = color[0]
    g_img[g_img < 169] = color[1]
    b_img[b_img < 169] = color[2]
    image = np.dstack([r_img,g_img,b_img])
    return image

if __name__ == '__main__':
    # 预测
    predict()
    # 结束
    ch = cv2.waitKey(0)
    while True:
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

