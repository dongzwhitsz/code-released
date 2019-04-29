'''
Date: 2019-04-10
Version: v0.0
Note:由于vscode的工作根目录为 E:/kaggle/src/
参考代码链接处：https://www.cnblogs.com/hellcat/p/8146748.html
将来自*.npy的numpy数组中的图片信息，实际将图片读出来，连同label和层次分类信息写入TFRecords格式的文件里面，
针对train valid数据使用相同的方式处理，对test数据仅保存图片信息，不含有分类信息。
，test数据集的label 使用字符串'None'代表。
其下所有的相对文件路径中的'.'均表示在工作目录这个根目录。
文件主要任务：将图片的图片信息和分类信息封装在tfrecords文件中，train, valid, test分别对应三个产生的tfrecords文件。
'''
import sys
import time
import tensorflow as tf
import numpy as np
import os
import cv2
import sys
from utils.v_timer import timer
import progressbar


def _int64_feature(value):
    """生成整数数据属性"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """生成字符型数据属性"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def data_transform_to_tfrecords(file_path):
    '''将数据从转换成tfrecords格式'''
    file_tfrecords_name = file_path[:-4] + '.tfrecords'
    file_list = np.load(file_path, encoding='bytes')
    file_len = len(file_list)
    i = 0
    with tf.io.TFRecordWriter(file_tfrecords_name) as writer:
        # 实时显示进度条
        widgets = [
            'Progress: ', progressbar.Counter(),
            ':{} '.format(file_len), progressbar.Percentage(), ' ',
            progressbar.Bar('#'), ' ', progressbar.Timer(),
            ' ', progressbar.ETA(), ' '
        ]
        bar = progressbar.ProgressBar(maxval=file_len, widgets=widgets)
        bar.start()
        for d in file_list:
            img = cv2.imread(d['img_id'])
            # 将图片进行缩小，如果不缩小，则磁盘无法放下，其缩小后的尺寸为 200x200x3
            img = cv2.resize(img, dsize=(200, 200))
            # img_width, img_height, _ = img.shape
            # 将图片的类型转换为bytes类型， tostring()函数为numpy矩阵的函数。
            img = img.tostring()
            level1 = d['level1']
            level2 = d['level2']
            level3 = d['level3']
            label = d['label']
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                # 存储图片的不同层次上的分类结果
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                'level': tf.train.Feature(int64_list=tf.train.Int64List(value=[level1, level2, level3]))
            }))
            try:
                writer.write(example.SerializeToString())  # 序列化为字符串,并将其写入磁盘文件
            except:
                # 如果在写的时候发生了写入异常，则将报错信息写入日志中，继续转换。报错一般是因为磁盘空间不足造成的。
                with open('./log/error_in_npz_to_tfrecords.txt', 'a') as f:
                    time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    img_info = 'img id: ' + d['img_id']
                    img_class = label
                    info = '*******************\n' + 'time:' + time_info + '\n' + img_info + '\nimg_class: ' + img_class + 'process: {}:{}\n'.format(i, file_len)
                    f.write(info)
            i += 1
            bar.update(i)
            if i == 1:
                # 将任务的启动信息写入日志记录中。
                with open('./log/npz_to_tfrecords_report.txt', 'a') as f:
                    time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    info = time_info + '\n' + file_tfrecords_name + '\nstart task: {}:{}\n\n'.format(i, file_len)
                    f.write(info)
        bar.finish()
        with open('./log/npz_to_tfrecords_report.txt', 'a') as f:
            time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            info = time_info + '\n' + file_tfrecords_name + '\n end task: {}:{}\n\n\n\n'.format(i, file_len)
            f.write(info)
    return


def test_data_transform_to_tfrecords(file_path):
    file_tfrecords_name = file_path[:-4] + '.tfrecords'
    file_list = np.load(file_path, encoding='bytes')
    file_len = len(file_list)
    i = 0
    with tf.io.TFRecordWriter(file_tfrecords_name) as writer:
        # 实时显示进度条
        widgets = [
            'Progress: ', progressbar.Counter(),
            ':{} '.format(file_len), progressbar.Percentage(), ' ',
            progressbar.Bar('#'), ' ', progressbar.Timer(),
            ' ', progressbar.ETA(), ' '
        ]
        bar = progressbar.ProgressBar(maxval=file_len, widgets=widgets)
        bar.start()
        for d in file_list:
            img = cv2.imread(d['img_id'])
            # 将图片进行缩小，如果不缩小，则磁盘无法放下，其缩小后的尺寸为 200x200x3
            img = cv2.resize(img, dsize=(200, 200))
            # img_width, img_height, _ = img.shape
            # 将图片的类型转换为bytes类型， tostring()函数为numpy矩阵的函数。
            img = img.tostring()
            id_bytes = bytes(d['img_id'], encoding='utf-8')
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_id':   tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_bytes])),
                'img_raw':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            try:
                writer.write(example.SerializeToString())  # 序列化为字符串,并将其写入磁盘文件
            except:
                # 如果在写的时候发生了写入异常，则将报错信息写入日志中，继续转换。报错一般是因为磁盘空间不足造成的。
                with open('./log/error_in_npz_to_tfrecords.txt', 'a') as f:
                    time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    img_info = 'img id: ' + d['img_id']
                    img_class = 'test'
                    info = '*******************\n' + 'time:' + time_info + '\n' + img_info + '\nimg_class: ' + img_class + 'process: {}:{}\n\n'.format(i, file_len)
            i += 1
            # print(i)
            if i == 1:
                # 将任务的启动信息写入日志记录中。
                with open('./log/npz_to_tfrecords_report.txt', 'a') as f:
                    time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    info = time_info + '\n' + file_tfrecords_name + '\nstart task: {}:{}\n\n'.format(i, file_len)
                    f.write(info)
            bar.update(i)
        bar.finish()
        with open('./log/npz_to_tfrecords_report.txt', 'a') as f:
            time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            info = time_info + '\n' + file_tfrecords_name + '\n end task: {}:{}\n\n\n\n'.format(i, file_len)
            f.write(info)
    return

train_npy_path = './data/train_image_on_level_classifications.npy'
valid_npy_path = './data/valid_image_on_level_classifications.npy'
test_npy_path = './data/test_image_on_level_classifications.npy'


def get_records():
    # 大约使用时间： 10024条记录，  587.851 s
    with timer('valid *.npy to valid *.tfrecords'):
        data_transform_to_tfrecords(valid_npy_path)

    # 大约使用时间： 996128条记录 10000条 / 15min, 共计用时 09：45：21
    with timer('train *.npy to train *.tfrecords'):
        data_transform_to_tfrecords(train_npy_path)

    # 大约使用时间： 89660条记录  共计用时 00:47:33
    with timer('test *.npy to test *.tfrecords'):
        test_data_transform_to_tfrecords(test_npy_path)


if __name__ == '__main__':
    ''' 打开相应的代码，则产生相应的文件覆盖 '''

    # # 大约使用时间： 10024条记录，  587.851 s
    # with timer('valid *.npy to valid *.tfrecords'):
    #     data_transform_to_tfrecords(valid_file_path)

    # 大约使用时间： 996128条记录 10000条 / 15min, 共计用时 09：45：21
    with timer('train *.npy to train *.tfrecords'):
        data_transform_to_tfrecords(train_npy_path)

    # # 大约使用时间： 89660条记录  共计用时 00:47:33
    # with timer('test *.npy to test *.tfrecords'):
    #   test_data_transform_to_tfrecords(test_npy_path)
