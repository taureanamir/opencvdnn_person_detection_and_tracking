"""
This class loads YOLOv4 model using OpenCV DNN and performs human detection 
and filtration based on the supplied region of interest (ROI).
"""


from datetime import datetime

from tensorflow.core.protobuf import config_pb2
import cv2
import os
import numpy as np
import logging
from pathlib import Path
import random
from config.config import config

class YOLOv4:
    def __init__(  self
                 , model_def
                 , weights_path
                 , conf_thres
                 , nms_thres
                 , img_size
                 , image=None
                 ):
        self.model_def = model_def
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.image = image
        self.model = None
        self.load_model()


    def load_model(self):
        # OpenCV DNN YOLOV4 
        net = cv2.dnn.readNet(self.weights_path, self.model_def)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        yolo = cv2.dnn_DetectionModel(net)
        yolo.setInputParams(size=(self.img_size, self.img_size), scale=1/255, swapRB=True)
        if self.image is not None:
            if self.image.any():       
                _c, _s, _b = yolo.detect(self.image, self.conf_thres, self.nms_thres)
        print("Loaded YOLOv4 model !!!")
        self.model = yolo


    def get_human_bboxes(self, detectionBoxes, roi=None):
        if roi:
            # 1. filter boxes that lie inside the ROI
            boxsInROI = []
            # print("boxs: {}".format(boxs))
            for box in detectionBoxes:
                # print("Detections bbox: {}".format(box))
                # box[0, 1, 2, 3] => topLeftX, topLeftY, width, height
                xmin = int(box[0])
                ymin = int(box[1])
                xmax = int(box[2]) + xmin
                ymax = int(box[3]) + ymin

                bbox_bottom_centerX, bbox_bottom_centerY = int((xmin + xmax)/2), ymax
                bbox_top_centerX, bbox_top_centerY = int((xmin + xmax)/2), ymin

                # check if the bbox bottom center lies inside the contour.
                # bboxInROI will be 0 or 1 if the bbox lies on the edge or inside the contour.
                offset = int(0.15 * box[3])  # add/subtract offset of 15% of the height of the bbox to the top/bottom of the Y coordinate of the bbox.
                bboxBottomInROI = cv2.pointPolygonTest(roi, (bbox_bottom_centerX, bbox_bottom_centerY), False)
                bboxTopInROI = cv2.pointPolygonTest(roi, (bbox_top_centerX, bbox_top_centerY), False)
                bboxBottomInROI2 = cv2.pointPolygonTest(roi, (bbox_bottom_centerX, bbox_bottom_centerY - offset), False)
                bboxTopInROI2 = cv2.pointPolygonTest(roi, (bbox_top_centerX, bbox_top_centerY + offset), False)

                if bboxTopInROI >= 0 or bboxBottomInROI >= 0 or bboxTopInROI2 >= 0 or bboxBottomInROI2 >= 0:
                    # print("Bbox inside ROI")
                    boxsInROI.append(box)

            return boxsInROI
        else:
            return detectionBoxes

    def get_humans(self, img):
        _classes, _scores, _boxes = self.model.detect(img, self.conf_thres, self.nms_thres)

        # ret_classes = []
        # ret_scores = []
        ret_boxes = []
        for (classid, score, box) in zip(_classes, _scores, _boxes):
            box = box.astype("int")
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)
            # only run for persons. person = 0
            if int(classid[0]) == 0:
                tbox0 = int(box[0])
                tbox1 = int(box[1])
                tbox2 = tbox0 + int(box[2])
                tbox3 = tbox1 + int(box[3])

                # self.draw_human_bbox(img, tbox0, tbox1, tbox2, tbox3)
                ret_boxes.append(box)
        # return ret_classes, ret_scores, ret_boxes
        return ret_boxes


    def draw_human_bbox(self, img, topLeftX, topLeftY, bottomRightX, bottomRightY):
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        cv2.rectangle(img, (topLeftX, topLeftY), (bottomRightX, bottomRightY), color=(b, g, r), thickness=2)
        # cv2.rectangle(img, (topLeftX, topLeftY), (topLeftX + len(label) * 20, top - 30), (b, g, r), cv2.FILLED)
        # cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255 - b, 255 - g, 255 - r), 1, cv2.LINE_AA)
        # cv2.rectangle(img, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (255, 255, 255), 1)
        # cv2.putText(img, (topLeftX, topLeftY), 0, 5e-3 * 200, (0, 255, 0), 1)