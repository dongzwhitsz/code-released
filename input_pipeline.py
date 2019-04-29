import tensorflow as tf
import progressbar
import numpy as np


def __parse_train_data(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            # 存储图片的不同层次上的分类结果
            'label':  tf.FixedLenFeature([], tf.int64),
            'level': tf.FixedLenFeature(shape=[3], dtype=tf.int64),
        })

    # 解析二进制数据格式,将之按照uint8格式解析
    img_raw = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    level = tf.cast(features['level'], tf.int32)

    # img  = tf.reshape(img_raw, shape=[200, 200, 3])
    img = tf.reshape(img_raw, shape=[200, 200, 3])
    img = tf.cast(img, tf.float32)
    img = tf.random_crop(img, [180, 180, 3])
    k = np.random.randint(3)
    img = tf.image.rot90(img, k)
    if k % 2 == 0:
        img = tf.image.random_flip_left_right(img)
    if k % 2 == 0:
        img = tf.image.random_flip_up_down(img)
    img = img / 128 - 1
    # img = tf.image.random_brightness(img, 0.0234375)
    return img, label, level


def __parse_valid_data(serialized_example):
    '''解析单个样例'''
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            # 存储图片的不同层次上的分类结果
            'label':  tf.FixedLenFeature([], tf.int64),
            'level': tf.FixedLenFeature(shape=[3], dtype=tf.int64),
        })

    # 解析二进制数据格式,将之按照uint8格式解析
    img_raw = tf.decode_raw(features['img_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    level = tf.cast(features['level'], tf.int32)

    img = tf.reshape(img_raw, shape=[200, 200, 3]) / 128 - 1
    img = tf.cast(img, tf.float32)
    img = tf.random_crop(img, [180, 180, 3])
    return img, label, level


def __parse_test_data(serialized_example):
    # 解析单个样例
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_id': tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string),
        })
    # 解析二进制数据格式,将之按照uint8格式解析
    # img_id = tf.decode_raw(features['img_raw'],tf.string)
    img_id = features['img_id']
    img_raw = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img_raw, shape=[200, 200, 3]) / 128 - 1
    img = tf.cast(img, tf.float32)
    img = tf.random_crop(img, [180, 180, 3])
    return img_id, img


# 用于读取train valid的数据
def read_data(tfrecords_file_path, batch_size, dataset_type=0):
    '''dataset_type: 0 for train set, 1 for validate set, 2 for test set'''
    dataset = tf.data.TFRecordDataset(tfrecords_file_path)
    if dataset_type is 0:
        dataset = dataset.map(__parse_train_data)
    elif dataset_type is 1:
        dataset = dataset.map(__parse_valid_data)
    else:
        dataset = dataset.map(__parse_test_data)
    dataset = dataset.shuffle(buffer_size=batch_size * 3).batch(batch_size).repeat()
    iterator = dataset.make_initializable_iterator()
    return iterator
