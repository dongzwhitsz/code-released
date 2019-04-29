'''
Date: 2019-04-09
Version: v0.0
Note:文件的主要任务为将提取原始文件的目录信息，将对应的图片的路径和其对应product tree上的分类信息提取出来 。
'''
import sys
import tensorflow as tf
import os
import json
from utils.v_timer import timer
import numpy as np
from train import get_config

config = get_config()
DATA_ROOT_PATH = ["DATA_ROOT_PATH"]
product_tree_path = os.path.join(DATA_ROOT_PATH, 'product_tree.json')


# 找到在product tree中特定的一个类的层次结构
def _get_labels_in_product_tree(tree_json, label_tag):
    # 在tree_json的字典里面，通过遍历查找对应的标签号：
    level1 = 0
    level2 = 0
    level3 = 0
    for level1_key, level1_dict in tree_json.items():
        for level2_key, level2_dict in level1_dict.items():
            for level3_key, level3_array in level2_dict.items():
                if label_tag in level3_array:
                    level1 = int(level1_key[-1]) - 1
                    level2 = int(level2_key[-1]) - 1
                    level3 = int(level3_key[-1]) - 1
                    return level1, level2, level3
    # 如果在product tree中遍历找不到该被查询的值， 则抛出ValueError异常
    raise ValueError('label_tag not found in the product tree')


def read_images_name_list(root_dir, product_tree_path=product_tree_path):
    '''
    为每一个来自train或者是valid的图片集的图片根据其在label，在product tree中找到其对应的层次分类，
    并连同最终分类的图片地址将其作为一个字典添加在file_list里面去；
    '''
    file_list = []

    # 读取产品树结构分类的json文件
    with open(product_tree_path) as f:
        tree_json = json.load(f)

    label_list = os.listdir(root_dir)
    for label in label_list:
        p = os.path.join(root_dir, label)
        img_list = os.listdir(p)
        for img in img_list:
            d = dict()
            d['img_id'] = os.path.join(p, img)
            if os.path.isdir(d['img_id']):
                continue
            if img == 'tmp_imgs.tfrecords':
                os.remove(d['img_id'])
                continue
            level1, level2, level3 = _get_labels_in_product_tree(tree_json, label)
            d['level1'] = level1
            d['level2'] = level2
            d['level3'] = level3
            d['label'] = label
            file_list.append(d)

    return file_list


def get_test_file_list(root_dir):
    '''
    由于test数据集其没有分类，特别为其加了一个数据读取的函数，其与train valid的条目的字典
    中，其默认分类的key相同，值为None
    '''
    file_list = []
    img_list = os.listdir(root_dir)
    for img in img_list:
        p = os.path.join(root_dir, img)
        d = dict()
        d['img_id'] = p
        d['level1'] = None
        d['level2'] = None
        d['level3'] = None
        d['label'] = None
        file_list.append(d)
    return file_list


train_path = os.path.join(DATA_ROOT_PATH, 'train')
valid_path = os.path.join(DATA_ROOT_PATH, 'val')
test_path = os.path.join(DATA_ROOT_PATH, 'test')


def get_npy():
    print('*' * 40)
    # 大约第一次400-500多s一个np.asarray()，加载进内存再次读取可能会40-50s一次
    with timer('turn the image dataset to a numpy array file'):
        file_list = np.asarray(read_images_name_list(root_dir=train_path))
    with timer('np.random.shuffle(file_list)'):
        np.random.shuffle(file_list)
    with timer('save the numpy array file'):
        np.save('./data/train_image_on_level_classifications', file_list)
    print('*' * 40)
    with timer('turn the image dataset to a numpy array file'):
        file_list = np.asarray(read_images_name_list(root_dir=valid_path))
    with timer('np.random.shuffle(file_list)'):
        np.random.shuffle(file_list)
    with timer('save the numpy array file'):
        np.save('./data/valid_image_on_level_classifications', file_list)
    print('*' * 40)
    with timer('test: turn the image dataset to a numpy array file'):
        file_list = np.asarray(get_test_file_list(root_dir=test_path))
    with timer('test: np.random.shuffle(file_list)'):
        np.random.shuffle(file_list)
    with timer('test: save the numpy array file'):
        np.save('./data/test_image_on_level_classifications', file_list)


if __name__ == '__main__':
    print('*' * 40)
    # 大约第一次400-500多s一个np.asarray()，加载进内存再次读取可能会40-50s一次
    with timer('turn the image dataset to a numpy array file'):
        file_list = np.asarray(read_images_name_list(root_dir=train_path))
    with timer('np.random.shuffle(file_list)'):
        np.random.shuffle(file_list)
    with timer('save the numpy array file'):
        np.save('./data/train_image_on_level_classifications', file_list)

    # print('*' * 40)
    # with timer('turn the image dataset to a numpy array file'):
    #     file_list = np.asarray(read_images_name_list(root_dir=valid_path))
    # with timer('np.random.shuffle(file_list)'):
    #     np.random.shuffle(file_list)
    # with timer('save the numpy array file'):
    #     np.save('./data/valid_image_on_level_classifications', file_list)

    # print('*' * 40)
    # with timer('test: turn the image dataset to a numpy array file'):
    #     file_list = np.asarray(get_test_file_list(root_dir=test_path))
    # with timer('test: np.random.shuffle(file_list)'):
    #     np.random.shuffle(file_list)
    # with timer('test: save the numpy array file'):
    #     np.save('./data/test_image_on_level_classifications', file_list)
