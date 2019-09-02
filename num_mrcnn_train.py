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
MODEL_DIR = os.path.join(ROOT_DIR, "model_train")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# The path of training images
# DATASET_ROOT_PATH = "/home/ubuntu/hjt/Mask_RCNN/_train_images/"
DATASET_ROOT_PATH = os.path.join(ROOT_DIR, "_train_images/")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

############################################################
#  Configurations
############################################################


class NUMConfig(Config):
    """Configuration for training on the toy NUM dataset.
    Derives from the base Config class and overrides values specific
    to the toy NUM dataset.
    """
    # Give the configuration a recognizable name
    NAME = "NUM"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # ● Number of classes (including background)
    NUM_CLASSES = 1 + 29  # background + myClass NUM

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
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
    # num_images = 100
    # batch_size = GPU_COUNT * IMAGES_PER_GPU
    # STEPS_PER_EPOCH = int(num_images / batch_size * (3 / 4))
    STEPS_PER_EPOCH = 60  # 50

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = STEPS_PER_EPOCH // (1000 // 50)
    VALIDATION_STEPS = 10

    # RPN_TRAIN_ANCHORS_PER_IMAGE = 256 // scale_max
    #
    # MINI_MASK_SHAPE = (56 // scale_max, 56 // scale_max)
    #
    # DETECTION_MAX_INSTANCES = 100 * scale_min * 2 // 3


# For show #
config = NUMConfig()
config.display()


class NUMDataset(utils.Dataset):
    """Generates the NUM synthetic dataset. The dataset consists of simple
    NUM (triangles, squares, circles) placed randomly on a blank surface.
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

    def load_NUM(self,count, img_floder, imglist):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # ● Add classes. We have only one class to add.
        self.add_class("NUM", 1, ".")
        self.add_class("NUM", 2, "0")
        self.add_class("NUM", 3, "1")
        self.add_class("NUM", 4, "2")
        self.add_class("NUM", 5, "3")
        self.add_class("NUM", 6, "4")
        self.add_class("NUM", 7, "5")
        self.add_class("NUM", 8, "6")
        self.add_class("NUM", 9, "7")
        self.add_class("NUM", 10, "8")
        self.add_class("NUM", 11, "9")
        self.add_class("NUM", 12, "-")
        self.add_class("NUM", 13, "%")
        self.add_class("NUM", 14, "才")
        self.add_class("NUM", 15, "基礎代謝量")
        self.add_class("NUM", 16, "筋肉量")
        self.add_class("NUM", 17, "男性")
        self.add_class("NUM", 18, "女性")
        self.add_class("NUM", 19, "内蔵脂肪")
        self.add_class("NUM", 20, "体内年齢")
        self.add_class("NUM", 21, "体脂肪率")
        self.add_class("NUM", 22, "生年月日")
        self.add_class("NUM", 23, "体重")
        self.add_class("NUM", 24, "身長")
        self.add_class("NUM", 25, "BMI")
        self.add_class("NUM", 26, "cm")
        self.add_class("NUM", 27, "kcal/日")
        self.add_class("NUM", 28, "kg")
        self.add_class("NUM", 29, "レベル")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of NUM sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            # print(imglist[i])
            # filestr = imglist[i].split(".")[0]
            filestr = imglist[i].split(".")[0]
            mask_path = DATASET_ROOT_PATH + filestr + "/label.png"
            yaml_path = DATASET_ROOT_PATH + filestr + "/info.yaml"

            # test path
            # print(DATASET_ROOT_PATH + filestr + "/img.png", 'img_path')
            # print(mask_path)
            # print(yaml_path)

            cv_img = cv2.imread(DATASET_ROOT_PATH + filestr + "/img.png")
            # plt.subplot(1, 1, 1), plt.title('test'), plt.imshow(cv_img)
            self.add_image("NUM", image_id=i, path=DATASET_ROOT_PATH + filestr + "/img.png",
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
        
    def load_mask(self, image_id):
        """Generate instance masks for NUM of the given image ID.
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
            if labels[i].find(".") != -1:
                labels_form.append(".")
            elif labels[i].find("0") != -1:
                labels_form.append("0")
            elif labels[i].find("1") != -1:
                labels_form.append("1")
            elif labels[i].find("2") != -1:
                labels_form.append("2")
            elif labels[i].find("3") != -1:
                labels_form.append("3")
            elif labels[i].find("4") != -1:
                labels_form.append("4")
            elif labels[i].find("5") != -1:
                labels_form.append("5")
            elif labels[i].find("6") != -1:
                labels_form.append("6")
            elif labels[i].find("7") != -1:
                labels_form.append("7")
            elif labels[i].find("8") != -1:
                labels_form.append("8")
            elif labels[i].find("9") != -1:
                labels_form.append("9")
            elif labels[i].find("-") != -1:
                labels_form.append("-")
            elif labels[i].find("%") != -1:
                labels_form.append("%")
            elif labels[i].find("才") != -1:
                labels_form.append("才")
            elif labels[i].find("基礎代謝量") != -1:
                labels_form.append("基礎代謝量")
            elif labels[i].find("筋肉量") != -1:
                labels_form.append("筋肉量")
            elif labels[i].find("男性") != -1:
                labels_form.append("男性")
            elif labels[i].find("女性") != -1:
                labels_form.append("女性")
            elif labels[i].find("内蔵脂肪") != -1:
                labels_form.append("内蔵脂肪")
            elif labels[i].find("体内年齢") != -1:
                labels_form.append("体内年齢")
            elif labels[i].find("体脂肪率") != -1:
                labels_form.append("体脂肪率")
            elif labels[i].find("生年月日") != -1:
                labels_form.append("生年月日")
            elif labels[i].find("体重") != -1:
                labels_form.append("体重")
            elif labels[i].find("身長") != -1:
                labels_form.append("身長")
            elif labels[i].find("BMI") != -1:
                labels_form.append("BMI")
            elif labels[i].find("cm") != -1:
                labels_form.append("cm")
            elif labels[i].find("kcal/日") != -1:
                labels_form.append("kcal/日")
            elif labels[i].find("kg") != -1:
                labels_form.append("kg")
            elif labels[i].find("レベル") != -1:
                labels_form.append("レベル")

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

    dataset_train = NUMDataset()
    dataset_train.load_NUM(count, DATASET_ROOT_PATH, imglist)
    dataset_train.prepare()

    dataset_val = NUMDataset()
    dataset_val.load_NUM(2, DATASET_ROOT_PATH, imglist)
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
                epochs=60,
                layers='all')

    train_end = time.time()
    print("***** The end time:", train_end)
    print("***** The training Time:.%s Seconds" % (train_end - train_start))
    os.system("sudo shutdown")


# nohup python num_mrcnn_train.py >> _logs/num_log_x.log 2>&1 &
# tensorboard -logdir=/home/ubuntu/hjt/digits_number/model_train/
# ssh -f -N -L 55555:localhost:6006 mars