#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Description: Yolov5 detect my version.
# Date: 2020/12/07
# Author: Steven Huang, Auckland, NZ

import argparse
import torch
import numpy as np
import cv2
import os

from download_weights import yolov5_path
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import time_sync
from utils.plots import Annotator, colors  # save_one_box
from models.common import DetectMultiBackend
from summary_model import summaryNet


# python yolov5_detect.py --weights .\weights\yolov5s.pt --source .\images\bus.jpg


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'./weights/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'./images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=r'./coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=r'./runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def preprocessImg(img, imgSize):
    img = letterbox(img, new_shape=imgSize)[0]
    # print('img.shape1:', img.shape, type(img), img.dtype)

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    device = 'cpu'
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # print('img.shape2:', img.shape, type(img), img.dtype, img.ndimension())
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # print('img.shape3:', img.shape, type(img), img.dtype, img.ndimension())
    return img


def predictImg(model, img0, opt, text_path=None):
    img = preprocessImg(img0, opt.imgsz)

    names = model.names
    # colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    # print('names:', len(names), names)

    t1 = time_sync()
    pred = model(img)
    # print('pred=', len(pred))
    # print('pred:', pred)
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                               classes=opt.classes, agnostic=opt.agnostic_nms,
                               max_det=opt.max_det)

    t2 = time_sync()

    view_img = True
    save_txt = True
    # save_crop = opt.save_crop

    for _, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # imc = img0.copy() if save_crop else img0  # for save_crop
        annotator = Annotator(img0, line_width=2, example=str(names))

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            s = ''
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt and text_path:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    # print('line=', line)
                    with open(text_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'  # '%s %.2f' % (names[int(cls)], conf)
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # img0 = save_one_box(xyxy, imc, 'out.jpg', BGR=True, save=False)

    # Print time (inference + NMS)
    print('%sDone. (%.3fs)' % (s, t2 - t1))
    return annotator.result()


def summary_model(model):
    print(model)
    summaryNet(model, (3, 640, 480))


def show_img(img):
    cv2.imshow('', img)
    cv2.waitKey(0)


def save_img(img, file=r'./images/out.jpg'):
    cv2.imwrite(file, img)


def load_model(weights=r'./weights/yolov5l.pt'):
    model = DetectMultiBackend(weights, device='cpu')
    # model = torch.load(w, map_location='cpu')['model'].float().fuse().eval()
    # print('type(model)=', type(model))
    # summary_model(model)
    return model


def main():
    print('yolov5_path=', yolov5_path)
    opt = parse_opt()
    print('opt=\n', opt)

    # w = r'./weights/yolov5l.pt'
    # imgFile = r'./images/zidane.jpg'
    w = opt.weights
    imgFile = opt.source

    model = load_model(w)

    head, tail = os.path.split(imgFile)
    dst = os.path.join(head, os.path.splitext(tail)[0] + '_pred.txt')

    img0 = cv2.imread(imgFile)
    img = predictImg(model, img0, opt, text_path=dst)

    show_img(img)

    dst = os.path.join(head, os.path.splitext(tail)[0] + '_out.jpg')
    save_img(img, dst)


if __name__ == '__main__':
    main()
