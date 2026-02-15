"""Project constants."""

from pathlib import Path

PROJECT_NAME = "GSRD"
PREDICTION_FORMAT_VERSION = "1.0.0"
DEFAULT_SEED = 20260215

COCO_VAL_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"

BDD100K_IMAGES_VAL_URL = "https://dl.cv.ethz.ch/bdd100k/data/100k_images_val.zip"
BDD100K_LABELS_URL = "https://dl.cv.ethz.ch/bdd100k/data/labels20_det.zip"

DEFAULT_WORKDIR = Path(".").resolve()
