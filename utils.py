import os

from six import b
import cv2
import logging
from logging import handlers
from config import config

class Utils:
    @staticmethod
    def overlay_framerate(frame, frame_rate):
        cv2.putText(frame, str(frame_rate) + " FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=1, lineType=1)


    @staticmethod
    def makeDir(_dir):
        directory = os.path.join(_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    @staticmethod
    def overlay_track(frame, track_id, bbox, _color):
        label = "Track: {}".format(track_id)

        b = _color[0][0]
        g = _color[0][1]
        r = _color[0][2]
        left, top, width, height = [int(i) for i in bbox]
        # print("left, top, width, height: {}, {}, {}, {}".format(left, top, width, height))
        cv2.rectangle(frame, bbox, color=(b, g, r), thickness=2)
        cv2.rectangle(frame, (left, top), (left + len(label) * 20, top - 30), (b, g, r), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_COMPLEX, 1, (255 - b, 255 - g, 255 - r), 1, cv2.LINE_AA)
                