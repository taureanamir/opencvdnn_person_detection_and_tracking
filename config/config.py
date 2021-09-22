import os
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

config = {
      "BASE_DIR": BASE_DIR
    , "INPUT_DIR": os.path.join(BASE_DIR, 'input')
    , "MODEL_DIR": os.path.join(BASE_DIR, 'models')
    , "CONFIG_DIR": os.path.join(BASE_DIR, 'config')
    , "LOGS_DIR": os.path.join(BASE_DIR, 'logs')
    , "TEST_DIR": os.path.join(BASE_DIR, 'test')
    , "MODELS": {
                     "YOLOv4_WEIGHTS": os.path.join(BASE_DIR, 'models', 'detector', 'yolov4.weights')
                    , "YOLOv4_CFG": os.path.join(BASE_DIR, 'models', 'detector', 'yolov4.cfg')
                    , "YOLOv4_IMG_SIZE": [608, 608] # width, height
                    , "DEEP_SORT": os.path.join(BASE_DIR, 'models', 'tracker', 'mars-small128.pb')
                }
    , "COCO_NAMES": os.path.join(BASE_DIR, 'models', 'detector', 'coco.names')
}


if __name__ == "__main__":
    for k, v in config.items():
        print("{}: {}".format(k, v))