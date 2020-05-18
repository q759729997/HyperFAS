import math
import cv2
import time
import numpy as np
import os

import keras
from keras.layers import *
import tensorflow as tf

from mtcnn import MTCNN

from keras.models import load_model

model = load_model("./model/fas.h5")


def load_mtcnn_model(model_path):
    mtcnn = MTCNN(model_path)
    return mtcnn


def test_one(X):
    TEMP = X.copy()
    X = (cv2.resize(X, (224, 224)) - 127.5) / 127.5
    t = model.predict(np.array([X]))[0]
    time_end = time.time()
    return t


def test_camera(mtcnn, index=0):

    cam = cv2.VideoCapture(0)

    while (1):
        success, frame = cam.read()
        image = frame

        img_size = np.asarray(image.shape)[0:2]

        bounding_boxes, scores, landmarks = mtcnn.detect(image)

        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            for det, pts in zip(bounding_boxes, landmarks):

                det = det.astype('int32')
                #print("face confidence: %2.3f" % confidence)
                det = np.squeeze(det)
                y1 = int(np.maximum(det[0], 0))
                x1 = int(np.maximum(det[1], 0))
                y2 = int(np.minimum(det[2], img_size[1] - 1))
                x2 = int(np.minimum(det[3], img_size[0] - 1))

                w = x2 - x1
                h = y2 - y1
                _r = int(max(w, h) * 0.6)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                x1 = cx - _r
                y1 = cy - _r

                x1 = int(max(x1, 0))
                y1 = int(max(y1, 0))

                x2 = cx + _r
                y2 = cy + _r

                h, w, c = frame.shape
                x2 = int(min(x2, w - 2))
                y2 = int(min(y2, h - 2))

                _frame = frame[y1:y2, x1:x2]
                score = test_one(_frame)

                print(score)
                if score > 0.95:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(1)


def test_photo(mtcnn, img_file, out_img_file):

    # cam = cv2.imread(img_file)
    image = cv2.imread(img_file)
    # print('shape:{}'.format(image.shape))

    img_size = np.asarray(image.shape)[0:2]

    bounding_boxes, scores, landmarks = mtcnn.detect(image)

    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        for det, pts in zip(bounding_boxes, landmarks):

            det = det.astype('int32')
            #  print("face confidence: %2.3f" % confidence)
            det = np.squeeze(det)
            y1 = int(np.maximum(det[0], 0))
            x1 = int(np.maximum(det[1], 0))
            y2 = int(np.minimum(det[2], img_size[1] - 1))
            x2 = int(np.minimum(det[3], img_size[0] - 1))

            w = x2 - x1
            h = y2 - y1
            _r = int(max(w, h) * 0.6)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            x1 = cx - _r
            y1 = cy - _r

            x1 = int(max(x1, 0))
            y1 = int(max(y1, 0))

            x2 = cx + _r
            y2 = cy + _r

            h, w, c = image.shape
            x2 = int(min(x2, w - 2))
            y2 = int(min(y2, h - 2))

            _frame = image[y1:y2, x1:x2]
            score = test_one(_frame)

            print(score, img_file)
            if score > 0.95:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # cv2.imshow("image", image)
    cv2.imwrite(out_img_file, image)


if __name__ == '__main__':

    mtcnn = load_mtcnn_model("./model/mtcnn.pb")
    # test_camera(mtcnn)
    image_names = [
        'qiaoyongtian_true_1',
        'qiaoyongtian_true_2',
        'lijiale_true_1',
        'lijiale_true_2',
        'zhaoshengao_true_1',
        'zhaoshengao_true_2',
        'qiaoyongtian_false_1',
        'qiaoyongtian_false_2',
        'lijiale_false_1',
        'lijiale_false_2',
        'zhaoshengao_false_1',
        'zhaoshengao_false_2',
    ]
    for image_name in image_names:
        img_file = os.path.join('./data/images', '{}.jpg'.format(image_name))
        out_img_file = os.path.join('./data/images', 'masked_{}.jpg'.format(image_name))
        test_photo(mtcnn, img_file, out_img_file)
    """
    [0.64997584] ./data/images\qiaoyongtian_true_1.jpg
    [0.19054134] ./data/images\qiaoyongtian_true_2.jpg
    [0.1899028] ./data/images\lijiale_true_1.jpg
    [0.9625011] ./data/images\lijiale_true_2.jpg
    [0.77500296] ./data/images\zhaoshengao_true_1.jpg
    [0.45846337] ./data/images\zhaoshengao_true_2.jpg

    [0.07987513] ./data/images\qiaoyongtian_false_1.jpg
    [0.31276345] ./data/images\qiaoyongtian_false_2.jpg
    [0.13304888] ./data/images\lijiale_false_1.jpg
    [0.00725103] ./data/images\lijiale_false_2.jpg
    [0.0028209] ./data/images\zhaoshengao_false_1.jpg
    [0.00350788] ./data/images\zhaoshengao_false_2.jpg
    """
