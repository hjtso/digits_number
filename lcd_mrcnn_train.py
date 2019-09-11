import os
import sys
import yaml
import math
import re
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "model_train")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# The path of training images
# DATASET_ROOT_PATH = "/Users/gsl/Desktop/Mask_RCNN-master/train_images_lcd/"
DATASET_ROOT_PATH = os.path.join(ROOT_DIR, "_train_images_lcd/")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

############################################################
#  Configurations
############################################################


class LCDConfig(Config):
    """Configuration for training on the toy LCD dataset.
    Derives from the base Config class and overrides values specific
    to the toy LCD dataset.
    """
    # Give the configuration a recognizable name
    NAME = "LCD"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 8
    IMAGES_PER_GPU = 1

    # ● Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + myClass LCD

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # Image size must be dividable by 2 at least 6 times to avoid fractions
    # when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc.
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    # scale_max = 1024 // IMAGE_MAX_DIM
    # scale_min = 1024 // IMAGE_MIN_DIM

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (32//scale_max, 64//scale_max, 128//scale_max, 256//scale_max, 512//scale_max)
    RPN_ANCHOR_SCALES = (8*6, 16*6, 32*6, 64*6, 128*6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 200 // scale_min
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    # num_images = 80
    # batch_size = GPU_COUNT * IMAGES_PER_GPU
    # STEPS_PER_EPOCH = int(num_images / batch_size * (3 / 4))
    STEPS_PER_EPOCH = 60

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = STEPS_PER_EPOCH // (1000 // 50)
    VALIDATION_STEPS = 10

    # RPN_TRAIN_ANCHORS_PER_IMAGE = 256 // scale_max
    #
    # MINI_MASK_SHAPE = (56 // scale_max, 56 // scale_max)
    #
    # DETECTION_MAX_INSTANCES = 100 * scale_min * 2 // 3


# For show #
config = LCDConfig()
config.display()


class LCDDataset(utils.Dataset):
    """Generates the LCD synthetic dataset. The dataset consists of simple
    LCD (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']

            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image,image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_LCD(self, count, imglist):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # ● Add classes. We have only one class to add.
        self.add_class("LCD", 1, "LCD")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of LCD sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            # print(imglist[i])
            filestr = imglist[i].split(".")[0]
            mask_path = DATASET_ROOT_PATH + filestr + "/label.png"
            yaml_path = DATASET_ROOT_PATH + filestr + "/info.yaml"

            # test path
            # print(DATASET_ROOT_PATH + filestr + "/img.png", 'img_path')
            # print(mask_path)
            # print(yaml_path)

            cv_img = cv2.imread(DATASET_ROOT_PATH + filestr + "/img.png")
            # plt.subplot(1, 1, 1), plt.title('test'), plt.imshow(cv_img)
            self.add_image("LCD", image_id=i, path=DATASET_ROOT_PATH + filestr + "/img.png",
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
        
    def load_mask(self, image_id):
        """Generate instance masks for LCD of the given image ID.
        """
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = self.from_yaml_get_class(image_id)
        labels_form = []

        # ● add class label
        for i in range(len(labels)):
            if labels[i].find("LCD") != -1:
                labels_form.append("LCD")

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


############################################################
#  Training
############################################################
if __name__ == '__main__':
    train_start = time.time()
    print("***** The start time:", train_start)

    img_floder = DATASET_ROOT_PATH
    imglist = os.listdir(img_floder)
    count = len(imglist)

    dataset_train = LCDDataset()
    dataset_train.load_LCD(count, imglist)
    dataset_train.prepare()

    dataset_val = LCDDataset()
    dataset_val.load_LCD(2, imglist)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    # Which weights to start with imagenet, coco, or last
    init_with = "coco"

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        # print(COCO_MODEL_PATH)
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head layers.
    # You can also pass a regular expression to select which layers to train by name pattern.
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=60,
    #             layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers.
    # You can also pass a regular expression to select which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='all')

    train_end = time.time()
    print("***** The end time:", train_end)
    print("***** The training Time:.%s Seconds" % (train_end - train_start))
    os.system("sudo shutdown")

# nohup python lcd_mrcnn_train.py >> _logs/lcd_log_x.log 2>&1 &
# tensorboard -logdir=/home/ubuntu/hjt/digits_number/model_train/
# ssh -f -N -L 55555:localhost:6006 mars
