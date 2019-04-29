import tensorflow as tf
import progressbar
from train import *
from input_pipeline import read_data


def validation(sess, model, config):
    model.purpose = 1
    right_num = 0
    valid_loss = 0
    VALID_SIZE = config["VALID_SIZE"]
    batch_size = config["batch_size"]

    valid_tfrecords_file_path = config["valid_file_path"]
    iterator = read_data(valid_tfrecords_file_path, batch_size, dataset_type=1)
    next_batch = iterator.get_next()
    sess.run(iterator.initializer)
    # 实时显示进度条
    widgets = [
            'valid: ', progressbar.Counter(),
            ':{} '.format(VALID_SIZE//batch_size),
            progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ',
            progressbar.Timer(), ' ', progressbar.ETA(), ' '
        ]
    bar = progressbar.ProgressBar(maxval=VALID_SIZE // batch_size, widgets=widgets)

    print('*' * 66)
    bar.start()
    for step in range(VALID_SIZE // batch_size):
        valid_image, valid_label, valid_level = sess.run(next_batch)
        batch_loss, batch_accuracy = sess.run(
            [model.loss, model.accuracy],
            feed_dict={
                model.image_input: valid_image,
                model.level_output: valid_level,
                model.label_output: valid_label
            }
        )
        right_num += int(batch_size * batch_accuracy)
        valid_loss += batch_loss
        bar.update(step)
    bar.finish()

    # 计算当前网络的在验证集上的精度和损失
    valid_accuracy = right_num / VALID_SIZE
    valid_loss = valid_loss / (VALID_SIZE // batch_size)

    # 把网络状态重新设置成训练状态
    model.purpose = 0
    return valid_accuracy, valid_loss
