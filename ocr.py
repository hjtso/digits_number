# -*- coding: UTF-8 -*-

import numpy as np
import cv2
import imutils
import os
import json
import glob
import time
from matplotlib import pyplot as plt
from scipy import stats
import skimage.io
import skimage.transform
from lcd_mrcnn import LCDMrcnn
from num_mrcnn import NUMMrcnn

HEIGHT = 500
SCORE = 0.666
CODE_FILE_SUCCESS = 200
CODE_FILE_NOTFOUND = 401


class Ocr:
    """OCR process.
    1. Determine whether there is a LCD.
    2. Determine the number in the LCD.
    Attributes:
        img: The image.
        resize_rate: Float. If resize the image, set this resize rate.(Default is 1)
        LCD_rectangle: Dict. The coordinatesthe of rectangle of LCD.
        LCD_tag: Boolean. Whether there is a LCD.
        lcd_mask_rcnn: LCDMrcnn. Mask-RCNN to detect LCD.
    """

    def __init__(self, img, lcd_mask_rcnn, num_mask_rcnn):
        self.img = img
        self.resize_rate = 1
        self.LCD_rectangle = None
        self.LCD_tag = False
        self.numbers = []
        self.lcd_mask_rcnn = lcd_mask_rcnn
        self.num_mask_rcnn = num_mask_rcnn

    def ocr_lcd_determine(self):
        """Determine whether there is a LCD.
        """
        scores, roi = self.lcd_mask_rcnn.test_image(self.img)
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

    def ocr_num_determine(self, cropped):
        """Determine whether there are 7-deg numbers, and get them.
        """
        number_list = self.num_mask_rcnn.test_image(cropped)
        return number_list

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
                               'result': {'LCD': None,
                                          'NUM': None}})

        # 1.LCD有無判定
        a = time.time()
        self.ocr_lcd_determine()
        b = time.time()
        print("● ● ● ● LCD判定 Time:.%s Seconds" % (b - a))

        if self.LCD_tag:
            print("→LCDあり。There is a LCD.")
        else:
            print("→LCDなし。There is not a LCD.")
            result_all = {'code': CODE_FILE_SUCCESS,
                          'message': "Upload is successful.",
                          'result': {'LCD': self.LCD_tag,
                                     'NUM': self.numbers}}
            return json.dumps(result_all)

        # 2.数字を取得する
        # image_cut = cv2.imread(self.img)
        image_cut = skimage.io.imread(self.img)
        cropped = image_cut[self.LCD_rectangle['top_left']['y']: self.LCD_rectangle['bottom_right']['y'],
                  self.LCD_rectangle['top_left']['x']: self.LCD_rectangle['bottom_right']['x']]  # [y0:y1, x0:x1]
        # cropped = skimage.transform.rescale(cropped, 0.25)
        # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        (filepath, tempfilename) = os.path.split(self.img)
        (filename, extension) = os.path.splitext(tempfilename)
        # cv2.imwrite("./_test_result/cut_{}.jpg".format(filename), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        skimage.io.imsave("./_test_result/cut_{}.jpg".format(filename))
        

        # TODO: Perspective Transform of LCD
        # reference: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

        # TODO: get the number from LCD
        a = time.time()
        self.numbers = self.ocr_num_determine(cropped)
        b = time.time()
        print("● ● ● ● NUM判定 Time:.%s Seconds" % (b - a))

        result_all = {'code': CODE_FILE_SUCCESS,
                      'message': "Upload is successful.",
                      'result': {'LCD': self.LCD_tag,
                                 'NUM': self.numbers}}

        return json.dumps(result_all, ensure_ascii=False)


if __name__ == '__main__':
    # test
    lcd_mask_rcnn = LCDMrcnn()
    num_mask_rcnn = NUMMrcnn()
    
    list_of_files = sorted(glob.glob('./_test_images/*.jpg'))
    for file in list_of_files:
        print("\nImage name:", file)
        ocr_img = Ocr(file, lcd_mask_rcnn, num_mask_rcnn)
        result = ocr_img.ocr_run()
        print("result:", result)