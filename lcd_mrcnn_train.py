import os
import sys
import random
import yaml
import math
import re
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import tensorflow as tf

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# The path of training images
# DATASET_ROOT_PATH = "/Users/gsl/Desktop/Mask_RCNN-master/images/receipt/"
DATASET_ROOT_PATH = os.path.join(ROOT_DIR, "images", "LCD/")

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
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # ● Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + myClass LCD

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # scale=1024//IMAGE_MAX_DIM
    # RPN_ANCHOR_SCALES = (32//scale, 64//scale, 128//scale, 256//scale, 512//scale)  # anchor side in pixels
    # RPN_ANCHOR_SCALES = (8*6, 16*6, 32*6, 64*6, 128*6)  # anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # scale = 1024 // IMAGE_MIN_DIM
    # TRAIN_ROIS_PER_IMAGE = 200 // scale
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    # num_images = 100
    # batch_size = GPU_COUNT * IMAGES_PER_GPU
    # STEPS_PER_EPOCH = int(num_images / batch_size * (3 / 4))
    STEPS_PER_EPOCH = 132

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = STEPS_PER_EPOCH // (1000 // 50)
    VALIDATION_STEPS = 7


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

    def load_LCD(self,count, img_floder, imglist):
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
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        # ● number of object
        count = 1
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


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
 
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


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
    dataset_train.load_LCD(count, DATASET_ROOT_PATH, imglist)
    dataset_train.prepare()

    dataset_val = LCDDataset()
    dataset_val.load_LCD(2, DATASET_ROOT_PATH, imglist)
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
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,   # 30
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers.
    # You can also pass a regular expression to select which layers to train by name pattern.
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=40,  # 30
    #             layers='all')

    train_end = time.time()
    print("***** The end time:", train_end)
    print("***** The training Time:.%s Seconds" % (train_end - train_start))
