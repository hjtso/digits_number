# -*- coding: UTF-8 -*-

import os
import json
from ocr import Ocr
from emrcnn import EMrcnn
from werkzeug.utils import secure_filename
from flask import Flask, request
app = Flask(__name__)
ALLOWED_EXTENSIONS = ['png', 'jpg']
UPLOAD_PATH = "pic"
CODE_FILE_NOTALLOWED = 402


def allowed_filename(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/pic', methods=['POST'])
def ocr():
    submitted_file = request.files['file']
    file_name = submitted_file.filename

    if submitted_file and allowed_filename(file_name):
        filename = secure_filename(file_name)
        directory = os.path.join(app.root_path, UPLOAD_PATH)
        if not os.path.exists(directory):
            os.mkdir(directory)

        # upload image
        file_path = os.path.join(directory, filename)
        submitted_file.save(file_path)
        print('Image Name: ', file_name)

        # analyze image
        ocr_img = Ocr(UPLOAD_PATH + "/" + file_name, e_mask_rcnn)
        result = ocr_img.ocr_run()

        # delete image
        os.remove(file_path)

        return result
    else:
        result = json.dumps({'code': CODE_FILE_NOTALLOWED,
                             'message': "Upload is failed. The file is not 'jpg' or 'png'.",
                             'LCD_coordinate': None,
                             'result': {'LCD': None}})

        return result


if __name__ == '__main__':
    e_mask_rcnn = EMrcnn()
    app.run(debug=True, host="0.0.0.0", port=10080)


# command eg.
# curl http://0.0.0.0:10080/pic -X POST -F file=@/Users/machen/Desktop/digits_number/test_images/1.jpg -s | python -m json.tool