# encoding:utf-8
import argparse
import json
import os
import os.path as osp
import base64
import warnings

import PIL.Image
import yaml

from labelme import utils

# import cv2
import numpy as np


# from skimage import img_as_ubyte

def json2point_txt(path_json, path_txt):  # 可修改生成格式
    with open(path_json, 'r') as path_json:
        jsonx = json.load(path_json)
        with open(path_txt, 'w+') as ftxt:
            for shape in jsonx['shapes']:
                label = str(shape['label']) + ' '
                xy = np.array(shape['points'])
                strxy = ''
                for m, n in xy:
                    m = int(m)
                    n = int(n)
                    # print('m:',m)
                    # print('n：',n)
                    strxy += str(m) + ' ' + str(n) + ' '
                label = strxy
                ftxt.writelines(label + "\n")


def json2gt_txt(path_json, path_img, path_labelimg, train_gt_txt):  # 可修改生成格式
    with open(path_json, 'r') as path_json:
        jsonx = json.load(path_json)
        with open(train_gt_txt, 'a+') as ftxt:
            labelarray = [0, 0, 0, 0]
            for shape in jsonx['shapes']:
                # label = int(shape['label'])
                if (shape['label'] == "left"):
                    label = 2
                elif (shape['label'] == "right"):
                    label = 3
                else:
                    label = 1
                # if label == 1:
                #     labelarray[0] = 1
                if label == 2:
                    labelarray[1] = 1
                if label == 3:
                    labelarray[2] = 1
                # if label == 4:
                #     labelarray[3] = 1

            labelstr = str(labelarray[0]) + " " + str(labelarray[1]) + " " + str(labelarray[2]) + " " + str(
                labelarray[3])
            ftxt.writelines(labelstr + "\n")


def main(map):
    # json_file = args.json_file
    data_root = f"../wrcg_data/{map}/"
    json_file = f"../wrcg_data/{map}/images/"
    dir_img = './images_dir/'
    dir_labelimg = './labels_dir/'
    train_gt_txt = data_root + 'list/train_gt.txt'
    list_path = os.listdir(json_file)
    # line_txt = './train_dataset/line_txt/' # txt存储目录
    list_dir = osp.join(data_root, 'list')
    if not osp.exists(list_dir):
        os.mkdir(list_dir)
    images_dir = osp.join(data_root, 'images_dir')
    if not osp.exists(images_dir):
        os.mkdir(images_dir)
    labels_dir = osp.join(data_root, 'labels_dir')
    if not osp.exists(labels_dir):
        os.mkdir(labels_dir)
        # if not os.path.exists(line_txt):
    #     os.makedirs(line_txt)

    label_name_to_value = {
        "background": 0,
        "left": 1,
        "right": 2,
    }

    for i in range(0, len(list_path)):
        if list_path[i].endswith('.json'):
            path_json = os.path.join(json_file, list_path[i])
            if os.path.isfile(path_json):
                data = json.load(open(path_json))
                img = utils.img_b64_to_arr(data['imageData'])
                lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

                lbl_viz = utils.draw_label(lbl, img, label_name_to_value)
                save_file_name = osp.basename(path_json).split('.')[0]

                PIL.Image.fromarray(img).save(osp.join(images_dir, '{}.jpg'.format(save_file_name)))
                PIL.Image.fromarray(lbl.astype(np.uint8)).save(osp.join(labels_dir, '{}.png'.format(save_file_name)))
                PIL.Image.fromarray(lbl_viz).save(osp.join(labels_dir, '{}_vis.png'.format(save_file_name)))

                # 生成lines.txt
                path_txt = images_dir + "/" + list_path[i].replace('.json', '.lines.txt')
                # print(path_txt)
                json2point_txt(path_json, path_txt)

                # 生成train_gt_txt
                path_img = dir_img + list_path[i].replace('.json', '.jpg')
                path_labelimg = dir_labelimg + list_path[i].replace('.json', '.png')

                with open(train_gt_txt, 'a+') as ftxt:
                    ftxt.writelines(path_img + " " + path_labelimg + " ")

                json2gt_txt(path_json, None, None, train_gt_txt)
            else:
                # 生成空的txt文件
                save_file_name = osp.basename(path_json).split('.')[0]
                path_txt = images_dir + "/" + list_path[i].replace('.json', '.lines.txt')
                with open(path_txt, 'w') as f:
                    pass

                # 生成全黑的label图像
                black_label = np.zeros((720, 1280), dtype=np.uint8)  # 假设图像大小为720x1280
                PIL.Image.fromarray(black_label).save(osp.join(labels_dir, '{}.png'.format(save_file_name)))
                
                # 生成train_gt_txt
                path_img = dir_img + list_path[i].replace('.json', '.jpg')
                path_labelimg = dir_labelimg + list_path[i].replace('.json', '.png')
                
                with open(train_gt_txt, 'a+') as ftxt:
                    ftxt.writelines(path_img + " " + path_labelimg + " ")
                    ftxt.writelines("0 0 0 0\n")  # 写入全0标签


if __name__ == '__main__':
    # base64path = argv[1]
    maps = ['portugal', 'sanremo', 'sardinia', 'turkey', 'wales']
    for map in maps:
        print(map)
        main(map)