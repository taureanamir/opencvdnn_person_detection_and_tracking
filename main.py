import cv2
import os
import numpy as np
import argparse
import sys

from config.config import config
from utils import Utils as utils

from tracker.deep_sort import nn_matching
from tracker.deep_sort.tracker import Tracker
from tracker.tools import generate_detections as gdet
from tracker.deep_sort import preprocessing
from tracker.deep_sort.detection import Detection
from detector.yolov4 import YOLOv4

import random
import time


def main(args):

    camera_id = "cam-" + args.camera_number
    
    INPUT_PATH = config['INPUT_DIR']
    LOGS_PATH = config['LOGS_DIR']
    
    # -------------------------------------------------------------------------------------------------------
    # Refactor model loading and warm up into separate methods
    # -------------------------------------------------------------------------------------------------------
    # detector model params
    YOLO_MODEL_CFG = config['MODELS']['YOLOv4_CFG']
    YOLO_MODEL_WT = config['MODELS']['YOLOv4_WEIGHTS']
    
    CONF_THRES = 0.8
    NMS_THRES = 0.4
    YOLO_IMG_SIZE = config['MODELS']['YOLOv4_IMG_SIZE'][0]

    # tracker model params
    TRACKER_MODEL = config['MODELS']['DEEP_SORT']
    tracker_encoder = gdet.create_box_encoder(TRACKER_MODEL, batch_size=1)
    print("Loaded DeepSORT tracking model !!!")
    MAX_COSINE_DISTANCE = 0.3
    NMS_MAX_OVERLAP = 1.0

    # Video stream to use
    source = "./test.mp4"

    video_cap = cv2.VideoCapture(source)

    if video_cap is None or not video_cap.isOpened():
        print("[ERROR] - ---------------------------------------------")
        print("[ERROR] - Input video stream {} doesn't exist".format(args.input))
        print("[ERROR] - ---------------------------------------------")
        sys.exit(1)

    # Tracker (Deep Sort)
    num_tracks = 0
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, nn_budget)
    tracker = Tracker(metric)
    boxes = None

    # initialize yolo detector model and warm it up
    yolo = YOLOv4(YOLO_MODEL_CFG, YOLO_MODEL_WT, CONF_THRES, NMS_THRES, YOLO_IMG_SIZE)

    while(True):
        start_time = time.time()
        ret, frame = video_cap.read()

        if ret == True:
            frame_num = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))

            humanBoxs = yolo.get_humans(frame)
            boxes = yolo.get_human_bboxes(humanBoxs, roi=None)

            # 2. extract features of these detections and only track these detections.
            features = tracker_encoder(frame, boxes)

            # score to 1.0 here.
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

            # Run non-maxima suppression.
            dboxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(dboxes, NMS_MAX_OVERLAP, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                tlwh = track.to_tlwh()
                tlwh = [int(i) for i in tlwh]
              
                # color for track
                if not track.color:
                    b = random.randint(0, 255)
                    g = random.randint(0, 255)
                    r = random.randint(0, 255)
                    track.color.append((b, g, r))


                utils.overlay_track(frame, track.track_id, tlwh, track.color)
            
            # Display the resulting frame    
            cv2.imshow('frame',frame)
            sys.stdout.flush()

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("----------------------------------------------------------")
                print("You requested to stop the program. Stopping gracefully !!!")
                print("----------------------------------------------------------")
                break

        # Break the loop
        else:
            break  

    # When everything done, release the video capture and video write objects
    video_cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print("----------------------------------------------------------")
    print("-------------------- Program ended !!! -------------------")
    print("----------------------------------------------------------")            


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Run main app')
    parser.add_argument('--camera_number', required=False, default="0",
                        help="Enter the camera id that you want to run the analytics for.")

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
