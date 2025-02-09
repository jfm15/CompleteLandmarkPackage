from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.DIR = ""
_C.DATASET.CACHE_DIR = ""
_C.DATASET.IMAGE_EXT = ""
_C.DATASET.DELIMITER = ","
_C.DATASET.USE_COLS = (0, 1)
_C.DATASET.KEY_POINTS = 0
_C.DATASET.CACHED_IMAGE_SIZE = []
_C.DATASET.PIXEL_SIZE = []
_C.DATASET.SIGMA = 1
_C.DATASET.GROUND_TRUTH_MULTIPLIER = 1.0
_C.DATASET.FLIP_AXIS = False
_C.DATASET.NO_OF_ANNOTATORS = 1
_C.DATASET.STATIONARY_POINT_DISTANCE = 0
_C.DATASET.STATIONARY_POINTS = []

_C.DATASET.AUGMENTATION = CN()
_C.DATASET.AUGMENTATION.REVERSE_AXIS = False
_C.DATASET.AUGMENTATION.FLIP = False
_C.DATASET.AUGMENTATION.FLIP_PAIRS = []
_C.DATASET.AUGMENTATION.ROTATION_FACTOR = 5
_C.DATASET.AUGMENTATION.INTENSITY_FACTOR = 0.25
_C.DATASET.AUGMENTATION.SF = 0.2
_C.DATASET.AUGMENTATION.TRANSLATION_X = 0.0
_C.DATASET.AUGMENTATION.TRANSLATION_Y = 0.0
_C.DATASET.AUGMENTATION.ELASTIC_STRENGTH = 50
_C.DATASET.AUGMENTATION.ELASTIC_SMOOTHNESS = 10

_C.TRAIN = CN()
_C.TRAIN.REPEATS = 1
_C.TRAIN.FRACTIONALISE_IMAGES = False
_C.TRAIN.ENSEMBLE_MODELS = 3
_C.TRAIN.LABELED_SUBSET = 30
_C.TRAIN.USE_UNLABELED = False
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.LR = 0.001
_C.TRAIN.EPOCHS = 30
_C.TRAIN.EARLY_STOPPING = 5
_C.TRAIN.FINE_TUNE_EPOCHS = 10
_C.TRAIN.FINAL_LAYER = "two_d_softmax"
_C.TRAIN.LOSS_FUNCTION = "nll_across_batch"


_C.VALIDATION = CN()
_C.VALIDATION.AGGREGATION_METHODS = []
_C.VALIDATION.SDR_AGGREGATION_METHOD = ""
_C.VALIDATION.SDR_THRESHOLDS = []
_C.VALIDATION.SAVE_IMAGE_PATH = ""
_C.VALIDATION.MEASUREMENTS_SUFFIX = ""
_C.VALIDATION.MEASUREMENTS = []
_C.VALIDATION.DIAGNOSES = []

_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'UnetPlusPlus'


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()