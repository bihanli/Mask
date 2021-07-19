import tensorflow as tf
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"]

colors = [(39, 129, 113), (164, 80, 133), (83, 122, 114), (99, 81, 172), (95, 56, 104), (37, 84, 86), (14, 89, 122),
          (80, 7, 65), (10, 102, 25), (90, 185, 109), (106, 110, 132), (169, 158, 85), (188, 185, 26), (103, 1, 17),
          (82, 144, 81), (92, 7, 184), (49, 81, 155), (179, 177, 69), (93, 187, 158), (13, 39, 73), (12, 50, 60),
          (16, 179, 33), (112, 69, 165), (15, 139, 63), (33, 191, 159), (182, 173, 32), (34, 113, 133), (90, 135, 34),
          (53, 34, 86), (141, 35, 190), (6, 171, 8), (118, 76, 112), (89, 60, 55), (15, 54, 88), (112, 75, 181),
          (42, 147, 38), (138, 52, 63), (128, 65, 149), (106, 103, 24), (168, 33, 45), (28, 136, 135), (86, 91, 108),
          (52, 11, 76), (142, 6, 189), (57, 81, 168), (55, 19, 148), (182, 101, 89), (44, 65, 179), (1, 33, 26),
          (122, 164, 26), (70, 63, 134), (137, 106, 82), (120, 118, 52), (129, 74, 42), (182, 147, 112), (22, 157, 50),
          (56, 50, 20), (2, 22, 177), (156, 100, 106), (21, 35, 42), (13, 8, 121), (142, 92, 28), (45, 118, 33),
          (105, 118, 30), (7, 185, 124), (46, 34, 146), (105, 184, 169), (22, 18, 5), (147, 71, 73), (181, 64, 91),
          (31, 39, 184), (164, 179, 33), (96, 50, 18), (95, 15, 106), (113, 68, 54), (136, 116, 112), (119, 139, 130),
          (31, 139, 34), (66, 6, 127), (62, 39, 2), (49, 99, 180), (49, 119, 155), (153, 50, 183), (125, 38, 3),
          (129, 87, 143), (49, 87, 40), (128, 62, 120), (73, 85, 148), (28, 144, 118), (29, 9, 24), (175, 45, 108),
          (81, 175, 64), (178, 19, 157), (74, 188, 190), (18, 114, 2), (62, 128, 96), (21, 3, 150), (0, 6, 95),
          (2, 20, 184), (122, 37, 185)]



MIXPRECISION = False
RANDOM_SEED = 1

# Parser
NUM_MAX_PAD = 100
THRESHOLD_POS = 0.5
THRESHOLD_NEG = 0.4
THRESHOLD_CROWD = 0.7

# Model
BACKBONE = "efficientnet-b2"
IMG_SIZE = 550
PROTO_OUTPUT_SIZE = 138
FPN_CHANNELS = 256
NUM_MASK = 32

# Loss
LOSS_WEIGHT_CLS = 1
LOSS_WEIGHT_BOX = 1.5
LOSS_WEIGHT_MASK = 6.125
LOSS_WEIGHT_SEG = 1
NEG_POS_RATIO = 3
MAX_MASKS_FOR_TRAIN = 100

# Detection
TOP_K = 200
CONF_THRESHOLD = 0.05
NMS_THRESHOLD = 0.5
MAX_NUM_DETECTION = 100

# Todo Add the training iteration for your dataset
TRAIN_ITER =800000

# Todo Add the number of classes in your dataset
NUM_CLASSES =len(COCO_CLASSES)+1

# Todo Design your own learning rate schedule
LR_STAGE = dict({'warmup_steps': 500,
             'warmup_lr': 1e-4,
             'initial_lr': 1e-3,
             'stages': [280000, 600000, 700000, 750000],
             'stage_lrs': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
})

# Todo Design your own anchors
ANCHOR = dict({"img_size": IMG_SIZE,
             "feature_map_size": [69, 35, 18, 9, 5],
             "aspect_ratio": [1, 0.5, 2],
             "scale": [24, 48, 96, 192, 384]
             })
backbones_objects = tf.keras.applications.EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                             include_top=False,
                             weights='imagenet')
backbones_extracted =['block3b_add', 'block4c_add', 'block6d_add']

parser_params = {
        "output_size": IMG_SIZE,
        "proto_out_size": PROTO_OUTPUT_SIZE,
        "num_max_padding": NUM_MAX_PAD,
        "augmentation_params": {
            # These are in RGB and for ImageNet
            "mean": (0.407, 0.457, 0.485),
            "std": (0.225, 0.224, 0.229),
            "output_size": IMG_SIZE,
            "proto_output_size": PROTO_OUTPUT_SIZE,
            "discard_box_width": 4. / float(IMG_SIZE),
            "discard_box_height": 4. / float(IMG_SIZE),
        },
        "matching_params": {
            "threshold_pos": THRESHOLD_POS,
            "threshold_neg": THRESHOLD_NEG,
            "threshold_crowd": THRESHOLD_CROWD
        },
        "label_map": {x+1: x+1 for x in range(len(COCO_CLASSES))}
        }
detection_params = {"num_cls": NUM_CLASSES,
        "label_background": 0,
        "top_k": TOP_K,
        "conf_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "max_num_detection": MAX_NUM_DETECTION
        }
loss_params = {
        "loss_weight_cls": LOSS_WEIGHT_CLS,
        "loss_weight_box": LOSS_WEIGHT_BOX,
        "loss_weight_mask": LOSS_WEIGHT_MASK,
        "loss_weight_seg": LOSS_WEIGHT_SEG,
        "neg_pos_ratio": NEG_POS_RATIO,
        "max_masks_for_train": MAX_MASKS_FOR_TRAIN
    }

model_params = {
        "backbone": BACKBONE,
        "fpn_channels": FPN_CHANNELS,
        "num_class": NUM_CLASSES,
        "num_mask": NUM_MASK,
        "anchor_params": ANCHOR,
        "detect_params": detection_params
    }

  