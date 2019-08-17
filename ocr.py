# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import imutils
import os
import json
import glob
import time
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from collections import Counter
from emrcnn import EMrcnn

HEIGHT = 500
GAUSSIANBLUR_KERNEL = 3
GAUSSIANBLUR_SIGMAX = 0  # the num should be odd
SCORE = 0.666
IMG_BORDER = 5
MIN_PERCENT = 0.10
CONTOUR_FOUND_NUM = 6  # contours of image, keep only the 6 largest one
EPSILON_PER = 0.026  # the approximated curve for epsilon = 0.026 of arc length.
QUADRILATERAL_POINT = 4
DRAW_CIRCLE_RADIUS = 6
BOUNDING_BORDER = 10
CODE_FILE_SUCCESS = 200
CODE_FILE_NOTFOUND = 401


class Ocr:
    """OCR process for receipt.
    1. Determine whether there is a LCD on the receipt.
    Attributes:
        img: The image of Receipt.
        resize_rate: Float. If resize the image, set this resize rate.(Default is 1)
        filter_option: Str. The filter to detect the edge of image. ("Contour" or "HSV")
        LCD_rectangle: Dict. The coordinatesthe of rectangle of LCD.
        receipt_rectangle: Dict. The coordinatesthe of rectangle of receipt which is cut.
        LCD_tag: Boolean. Whether there is a LCD on the receipt.
        quadrilateral_tag: Boolean. Whether there is a quadrilateral contour on the receipt image.
        no_obstacle_tag: Boolean. Whether there are obstacles on the receipt image.
        e_mask_rcnn: EMrcnn. Mask-RCNN to detect LCD.
    """

    def __init__(self, img, e_mask_rcnn):
        self.img = img
        self.resize_rate = 1
        self.LCD_rectangle = None
        self.LCD_tag = False
        self.e_mask_rcnn = e_mask_rcnn

    def ocr_LCD_determine(self):
        """Determine whether there is a LCD on the receipt.
        """
        scores, roi = self.e_mask_rcnn.test_image(self.img)
        if scores is not None:
            scores = scores.tolist()
            num = scores.index(max(scores))
            [y1, x1, y2, x2] = roi[num]
            self.LCD_rectangle = {'top_left': {'x': int(x1), 'y': int(y1)},
                                  'bottom_right': {'x': int(x2), 'y': int(y2)}}

            # attention: there maybe multi LCDs found in image. Use the max one.
            if max(scores) > SCORE:
                self.LCD_tag = True

        # For show #  Draw the rectangle of LCD
        # image_test = cv2.imread(self.img)
        # image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
        # cv2.rectangle(image_test, (x1, y1), (x2, y2), (255, 0, 255), 2)
        # plt.subplot(1, 1, 1), plt.title('original'), plt.imshow(image_test)
        # plt.show()

    def ocr_run(self):
        """OCR run
        Returns:
            json: The result of OCR test.
        """
        try:
            if not os.path.exists(os.path.join(self.img)):
                raise ValueError
        except ValueError:
            print("<ERROR> The file doesn't exist.")
            return json.dumps({'code': CODE_FILE_NOTFOUND,
                               'message': "The file can not be found in server",
                               'LCD_coordinate': None,
                               'result': {'LCD': None}})

        # 1.自署有無判定
        a = time.time()
        self.ocr_LCD_determine()
        b = time.time()
        print("● ● ● ● LCD判定 Time:.%s Seconds" % (b - a))

        if self.LCD_tag:
            print("→LCDあり。There is a LCD.")
        else:
            print("→LCDなし。There is not a LCD.")

        # Cut image
        image_cut = cv2.imread(self.img)
        cropped = image_cut[self.LCD_rectangle['top_left']['y']: self.LCD_rectangle['bottom_right']['y'],
                  self.LCD_rectangle['top_left']['x']: self.LCD_rectangle['bottom_right']['x']]  # [y0:y1, x0:x1]
        cv2.imwrite("./test_result/cropped.jpg", cropped)

        # For show #
        # _image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.subplot(2, 2, 4), plt.title('result'), plt.imshow(_image)
        # plt.show()

        result_all = {'code': CODE_FILE_SUCCESS,
                      'message': "Upload is successful.",
                      'LCD_coordinate': self.LCD_rectangle,
                      'result': {'LCD': self.LCD_tag}}

        return json.dumps(result_all)


if __name__ == '__main__':
    # test
    e_mask_rcnn = EMrcnn()
    list_of_files = sorted(glob.glob('/Users/machen/Desktop/digits_number/test_images/*.jpg'))
    for file in list_of_files:
        print("\nImage name:", file)
        ocr_img = Ocr(file, e_mask_rcnn)
        result = ocr_img.ocr_run()
