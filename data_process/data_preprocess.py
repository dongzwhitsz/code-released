import sys
from model.model_multi_task import ModelMultiTask
import numpy as np
import pandas as pd
import tensorflow as tf
import progressbar
import time
import os
import cv2
from train import get_config


def __parse_example(serialized_example):
    # 解析单个样例
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_id': tf.FixedLenFeature([], tf.string),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })
    img_id = features['img_id']
    img_raw = tf.decode_raw(features['img_raw'], tf.uint8)
    # width = tf.cast(features['width'],tf.int32)
    # height = tf.cast(features['height'],tf.int32)
    img = tf.reshape(img_raw, shape=[200, 200, 3])
    return img_id, img


def data_preprocess(img_root_path, sess, label, model, image_input, batch_size, saver_path):
    imgs_name_lists = os.listdir(img_root_path)
    # imgs_path_lists = [os.path.join(img_root_path, i) for i in imgs_name_lists]
    imgs_path_lists = imgs_name_lists
    tmp_tfrecords_file_name = os.path.join(img_root_path, 'tmp_imgs.tfrecords')
    # 实时显示进度条
    widgets = [
        'label_{}: '.format(label), progressbar.Counter(),
        ':{} '.format(len(imgs_path_lists)), progressbar.Percentage(),
        ' ', progressbar.Bar('#'), ' ', progressbar.Timer(),
        ' ', progressbar.ETA(), ' '
        ]
    bar = progressbar.ProgressBar(maxval=len(imgs_path_lists), widgets=widgets)
    print('***************** begin turn the image to tfrecords ******************')
    bar.start()

    with tf.io.TFRecordWriter(tmp_tfrecords_file_name) as writer:
        for i, img_path in enumerate(imgs_path_lists):
            img = cv2.imread(os.path.join(img_root_path, img_path))
            img = cv2.resize(img, dsize=(200, 200))
            # width, height, _ = img.shape
            img = img.tostring()
            id_bytes = bytes(img_path, encoding='utf-8')
            example = tf.train.Example(features=tf.train.Features(feature={
                'img_id':   tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_bytes])),
                'img_raw':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            }))
            writer.write(example.SerializeToString())  # 序列化为字符串,并将其写入磁盘文件
            bar.update(i)
        bar.finish()
    tmp_dataset = tf.data.TFRecordDataset(tmp_tfrecords_file_name)
    tmp_dataset = tmp_dataset.map(__parse_example)
    tmp_dataset = tmp_dataset.repeat().batch(batch_size)
    iterator = tmp_dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    sess.run(iterator.initializer)
    widgets = [
        'label_{}: '.format(label), progressbar.Counter(),
        ':{} '.format(len(imgs_path_lists)), progressbar.Percentage(),
        ' ', progressbar.Bar('#'), ' ',
        progressbar.Timer(), ' ', progressbar.ETA(), ' '
        ]
    bar = progressbar.ProgressBar(maxval=len(imgs_path_lists), widgets=widgets)
    print('***************** begin turn the predict the images ******************')
    bar.start()
    d = dict()
    for i in range(len(imgs_path_lists) // batch_size + 1):
        img_id, img = sess.run(next_batch)
        _label = img_id
        softmax = sess.run(
            model.softmax4,
            feed_dict={
                image_input: img,
            }
        )

        t = np.array(softmax[:, label])
        softmax = t.flatten()
        for i, la in enumerate(_label):
            d[la] = softmax[i]
            bar.update(i)
    bar.finish()

    sorted_softmax = sorted(d, key=lambda x: d[x])

    delete_dir = os.path.join(img_root_path, 'should_delete')
    if not os.path.exists(delete_dir):
        os.mkdir(delete_dir)
    del_num = int(len(sorted_softmax) * 0.05)
    for img in sorted_softmax[:del_num]:
        img = os.path.join(img_root_path, str(img, encoding='utf-8'))
        src_img = cv2.imread(img)
        dst_img = os.path.join(delete_dir, img.split('\\')[-1])
        im = cv2.imwrite(dst_img, src_img)
        os.remove(img)

    # 删除tfrecords文件
    os.remove(tmp_tfrecords_file_name)
    return

# 20:36 -
if __name__ == '__main__':
    config = get_config()
    batch_size = config["batch_size"]
    base_learning_rate = config["base_learning_rate"]
    keep_prob = config["keep_prob"]
    image_input = tf.placeholder(dtype=tf.float32, shape=[None, 200, 200, 3])
    model = ModelMultiTask(base_learning_rate, keep_prob)
    softmax1, softmax2, softmax3, softmax4 = model.inference(image_input)
    saver = tf.train.Saver()
    saver_path = config["final_model_path"]
    DATA_ROOT_PATH = config["DATA_ROOT_PATH"]
    with tf.Session() as sess:
        checkpoint = tf.train.latest_checkpoint(saver_path)
        if checkpoint is None:
            raise ValueError('no model found')
        else:
            saver.restore(sess, checkpoint)
            model.is_train = False
        for i in range(1900, 2019):
            print('******************* label: {} *********'.format(i))
            img_root_path = os.path.join(DATA_ROOT_PATH, 'train', str(i))
            data_preprocess(img_root_path, sess, i, model, image_input, saver_path=saver_path)
