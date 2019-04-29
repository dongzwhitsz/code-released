import tensorflow as tf
import sys
from model import *
import valid
import progressbar
import time
from input_pipeline import read_data
import json
import os
import threading
import time
from utils import *


def __auto_change_learning_rate(epoch):
    global config
    if config['learning_rate'] > 0:
        return config['learning_rate']
    else:
        lr = 0.1
        if epoch < 2:
            lr = 0.1
        elif epoch < 4:
            lr = 0.06
        elif epoch < 6:
            lr = 0.01
        elif epoch < 10:
            lr = 0.005
        elif epoch < 13:
            lr = 0.002
        elif epoch < 20:
            lr = 0.001
        else:
            lr = max(0.001 / (2 ** (epoch - 14)), 1e-7)
        return lr


def train(restored_path='./checkpoint/saver', restore=True):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global config
    valid_best_accuracy = 0.0
    final_model_path = config["final_model_path"]
    batch_size = config["batch_size"]
    train_tfrecords_file_path = config["train_file_path"]

    valid_checkpoint = tf.train.latest_checkpoint(final_model_path)
    if valid_checkpoint is not None:
        valid_best_accuracy = float(valid_checkpoint.split('_')[2])

    iterator = read_data(train_tfrecords_file_path, batch_size, dataset_type=0)
    next_batch = iterator.get_next()

    # 创建一个模型， 搭建静态图
    model_name = config["model"]
    keep_prob = config["keep_prob"]

    model = get_model(model_name, purpose=0)

    saver = tf.train.Saver(max_to_keep=3)
    final_model_saver = tf.train.Saver(max_to_keep=3)

    # 用于tensorboard中的scalar的显示的张量
    tf.summary.scalar('train_loss', model.loss)
    tf.summary.scalar('train_accuracy', model.accuracy)
    valid_accuracy, valid_loss = 0.0, 0.0
    merged_summary = tf.summary.merge_all()

    tensorboard_root_dir = config["tensorboard_root_dir"]
    saver_root_dir = config["saver_root_dir"]
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        tensorbaord_writer = tf.summary.FileWriter(tensorboard_root_dir, session=sess)
        tensorbaord_writer.add_graph(sess.graph)

        # 判断从存储的模型中恢复数据继续训练，还是重新进行训练。
        start_epoch = 0
        if restore is True:
            checkpoint = tf.train.latest_checkpoint(restored_path)
            # 如果存在可以恢复的模型数据， 则直接恢复
            if checkpoint:
                saver.restore(sess, checkpoint)
                print('*********************train is restored from {}*******************'.format(restored_path))
                start_epoch = int(checkpoint.split('_')[1])
            else:
                print('***********restored failed , initialized from the default setting********')
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
        else:
            print('****************initialized from the default setting************')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

        sess.run(iterator.initializer)
        epochs = config["epochs"]
        log_step = config["log_step"]
        TRAIN_SIZE = config["TRAIN_SIZE"]

        for epoch in range(start_epoch, epochs):
            batch_size = config["batch_size"]
            print('**********training——epoch num:{},batch_size:{}, learning_rate:{} ********'.format(epoch, batch_size, config["learning_rate"]))
            # 实时显示进度条
            widgets = [
                'train:lr:{} bs:{}: '.format(__auto_change_learning_rate(epoch), batch_size), progressbar.Counter(),
                ':{} '.format(TRAIN_SIZE//batch_size),
                progressbar.Percentage(),
                ' ', progressbar.Bar('#'), ' ',
                progressbar.Timer(),
                ' ', progressbar.ETA(), ' '
            ]
            bar = progressbar.ProgressBar(maxval=TRAIN_SIZE//batch_size, widgets=widgets, term_width=120)
            bar.start()

            for step in range(TRAIN_SIZE // batch_size + 1):
                train_image, train_label, train_level = sess.run(next_batch)
                _, train_loss, global_step, s, softmax_v = sess.run(
                    [
                        model.train_op, model.loss, model.global_step, merged_summary, model.softmax_v
                    ],
                    feed_dict={
                        model.learning_rate: __auto_change_learning_rate(epoch),
                        model.weights_regularize_lambda: config["weights_regularize_lambda"],
                        model.three_class_regularize_lambda: config["three_class_regularize_lambda"],
                        model.keep_prob: config["keep_prob"],
                        model.lambda_level1: config["lambda_level1"],
                        model.lambda_level2: config["lambda_level2"],
                        model.lambda_level3: config["lambda_level3"],
                        model.lambda_label: config["lambda_label"],   
                        model.image_input: train_image,
                        model.level_output: train_level,
                        model.label_output: train_label
                    }
                )
                bar.update(step)

                if step % 300 == 0 and step != 0:
                    saver_name = os.path.join(saver_root_dir, 'epoch_{}_step_{}.ckpt'.format(epoch, step))
                    saver.save(sess, saver_name, global_step=global_step)

                # Tensorboard可视化数据
                if step % config["log_step"] == 0:
                    tensorbaord_writer.add_summary(s, global_step)

                valid_interval = TRAIN_SIZE // batch_size // 4
                try_to_validate = config['try_to_validate']
                if try_to_validate is True or step % valid_interval is 0 and step is not 0:
                    # 结束之后做一次验证集的测试
                    valid_accuracy, valid_loss = valid.validation(sess, model, config)
                    valid_summary = tf.Summary()
                    valid_summary.value.add(tag='valid_loss', simple_value=valid_loss)
                    tensorbaord_writer.add_summary(valid_summary, global_step)
                    valid_summary = tf.Summary()
                    valid_summary.value.add(tag='valid_accuracy', simple_value=valid_accuracy)
                    tensorbaord_writer.add_summary(valid_summary, global_step)

                    with open('./log/valid_reports.txt', 'a') as f:
                        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        train_log = 'train global_step: {}\n'.format(global_step)
                        log_info = 'valid epoch_{}_{}: accuracy: {}, loss: {} \n'.format(epoch, step//valid_interval, valid_accuracy, valid_loss)
                        log = '\n' + time_info + '\n' + log_info + train_log + '\n'
                        f.write(log)

                    if valid_accuracy > valid_best_accuracy:
                        valid_best_accuracy = valid_accuracy
                        saver_name = os.path.join(final_model_path, 'accuracy_{:.5f}_epoch_{}_final.ckpt'.format(valid_best_accuracy, epoch))
                        final_model_saver.save(sess, saver_name, global_step=global_step)
            bar.finish()


def update_config():
    global lock
    global config
    while True:
        lock.acquire()
        config = json.load(open('./config.json'))
        time.sleep(2)
        lock.release()


config = {}
lock = threading.Lock()


def run():
    global lock
    global config
    thread_update_config = threading.Thread(target=update_config)
    thread_update_config.setDaemon(True)
    thread_update_config.start()
    time.sleep(2)
    thread_train = threading.Thread(target=train, args=(config['restored_path'], True))
    thread_train.start()


if __name__ == "__main__":
    run()
