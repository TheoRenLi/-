# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from entity_detection.diction_tool import object2id
from entity_detection.model import *
import re

import tools.infer.utility as utility
from ppocr.utils.utility import initial_logger

logger = initial_logger()
import cv2
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
import copy
import numpy as np
import pandas as pd

import math
import time
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from PIL import Image
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt



class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args['use_angle_cls']
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        # print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            # print("cls num  : {}, elapse : {}".format(
                # len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        # print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    # 载入配置文件
    path_model = 'entity_detection/'
    config_entity_classfy = torch.load(path_model + 'State.pth', map_location='cpu')
    
    # 初始化编码对象
    things2id = object2id()
    things2id.add_obj_id('char', path_model + 'vocab.txt')
    things2id.add_obj_id('entity_classfy_label', config_entity_classfy['entity_classfy_label'])
    
    # 模型初始化和加载
    dic_size, embed_dim, hidden_size, target_size = config_entity_classfy['model_dim']
    model = LAnn_prediction(dic_size, embed_dim, hidden_size, target_size)
    model.load_state_dict(config_entity_classfy['state'])

    image_file_list = get_image_file_list(args['image_dir'])
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args['vis_font_path']
    result = dict()
    img_name_list = list()
    predict = list()
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = Image.open(image_file).convert('RGB')
            img = np.copy(np.array(img)[:,:,::-1])
            h,w,c = img.shape
            max_l = max([h,w])
            scale = 1060 / max_l
            img = cv2.resize(img,(int(scale*w), int(scale*h)))
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)

        print('='*30)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        w,h = image.size
        txts = []
        boxes = []
        for i in range(len(rec_res)):
            score = rec_res[i][1]
            txt = rec_res[i][0]
            box = dt_boxes[i]
            if score >= 0.5:
                txts.append(txt)
                boxes.append(box)
        center_xs = [box[0][0] + 0.5 * (box[1][0] - box[0][0]) for box in boxes] # 根据box计算中心坐标center_x
        center_x_dist_coefs = [1 - abs(0.5 * w - center_x) / (0.5 * w) for center_x,box in zip(center_xs,boxes)] # 根据中心坐标center_x计算中心距权重
        areas = [c * (box[1][0] - box[0][0]) * (box[3][1] - box[0][1]) / (len(t) + 1) for c,box,t in zip(center_x_dist_coefs,boxes,txts)] # 根据box计算面积，并用中心距权重进行调节面积
        areas_total = sum(areas)
        env = ''.join(txts)
        txts_new = []
        areas_new = []
        for txt, area in zip(txts, areas):# 预处理：排除识别出的特殊字符（逗号，句号）
            if len(txt.strip()) == 0:
                confidence.append(0)
                continue
            symbol = re.search(r"\W", txt)
            if symbol != None:
                txt_splited = txt.split(txt[symbol.start()])
                txt_splited = [t for t in txt_splited if len(t.strip())!=0]
                txts_new += txt_splited
                areas_new += [area]*len(txt_splited)
                continue
            else:
                txts_new.append(txt)
                areas_new.append(area)
        confidence = []

        for txt,area in zip(txts_new, areas_new):# 结合图像特征与语义特征计算置信度
            txt_tensor = torch.tensor(things2id.obj_2_id('char', txt))
            env_tensor = torch.tensor(things2id.obj_2_id('char', env))
            logists = model([txt_tensor.unsqueeze(0),env_tensor.unsqueeze(0)])
            
            text_prob,arg = torch.max(torch.softmax(logists, 1), 1)
            label = things2id.id_2_obj('entity_classfy_label', arg.item())
            text_prob = text_prob.item() if '1_' in label else -0.2 * text_prob.item()
            area_prob = area / areas_total
            confidence.append(0.2 * text_prob + 0.8 * area_prob) # 总置信度为文字语义概率与图像特征概率加权求和

        if confidence==[]:
            print('Not found!')
            predict.append('Not found')
        else:    
            # 1.选出置信度最大的OCR结果
            # 2.判断最大置信度是否超过阈值
            # 3.输出结果
            max_confidence = max(confidence)
            if max_confidence > 0.3:
                print('企业实体:')
                print('\t' + txts_new[confidence.index(max_confidence)])
                print('\tConfidence {}'.format(max_confidence))
                predict.append(txts_new[confidence.index(max_confidence)])
            else:
                print('Not found!')
                predict.append('Not found')
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))
        img_name_list.append(os.path.split(image_file)[-1])
    # 保存模型
    result['图片'] = img_name_list
    result['商铺名称'] = predict
    df = pd.DataFrame(result, columns=['图片', '商铺名称'])
    df.to_excel('./result.xlsx', index=None)

if __name__ == "__main__":
    # 输入图片路径(图片或文件夹)
    main(utility.parse_args())
    os.system("pause")
