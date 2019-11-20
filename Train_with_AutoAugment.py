# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import time
import argparse

import numpy as np
import tensorflow as tf

import AutoAugment.AutoAugment as auto_augment

from WideResNet import *
from DataAugmentation import *

from Define import *
from Utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

learning_rate = INIT_LEARNING_RATE

ckpt_path = './model/WideResNet_with_AutoAugment.ckpt'
log_txt_path = './log_with_AutoAugment.txt'
tensorboard_path = './logs/train_with_AutoAugment'

log_print('# learning rate : {}'.format(learning_rate), log_txt_path)
log_print('# batch size : {}'.format(BATCH_SIZE), log_txt_path)
log_print('# max_iteration : {}'.format(MAX_ITERATION), log_txt_path)

# 1. dataset
train_data_list = np.load('./dataset/train.npy', allow_pickle = True)
test_data_list = np.load('./dataset/test.npy', allow_pickle = True)
test_iteration = len(test_data_list) // BATCH_SIZE

# 2. model
shape = [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL]

x_var = tf.placeholder(tf.float32, [BATCH_SIZE] + shape, name = 'image/labeled')
x_label_var = tf.placeholder(tf.float32, [BATCH_SIZE, CLASSES])
is_training = tf.placeholder(tf.bool)

logits_op, _ = WideResNet(x_var, is_training)

# with ema
train_vars = tf.get_collection('trainable_variables', 'Wider-ResNet-28')

# calculate Loss_x, Loss_u
loss_x_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits_op, labels = x_label_var)
loss_x_op = tf.reduce_mean(loss_x_op)

loss_op = loss_x_op

ema = tf.train.ExponentialMovingAverage(decay = EMA_DECAY)
ema_op = ema.apply(train_vars)

_, predictions_op = WideResNet(x_var, is_training, getter = get_getter(ema))

l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(x_label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

# 3. optimizer & tensorboard 
learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op, colocate_gradients_with_ops = True)
    train_op = tf.group(train_op, ema_op)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/Labeled_Loss' : loss_x_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,                                                                                                                                                                                                                                                     
    'Accuracy/Train' : accuracy_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

test_accuracy_var = tf.placeholder(tf.float32)
test_accuracy_op = tf.summary.scalar('Accuracy/Test', test_accuracy_var)

# 4. train loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
train_writer = tf.summary.FileWriter(tensorboard_path)

train_ops = [train_op, loss_op, loss_x_op, l2_reg_loss_op, accuracy_op, train_summary_op]

loss_list = []
x_loss_list = []
l2_reg_loss_list = []
accuracy_list = []
train_time = time.time()

for iter in range(1, MAX_ITERATION + 1):
    if iter in DECAY_ITERATION:
        learning_rate /= 10.
    
    np.random.shuffle(train_data_list)
    batch_data_list = train_data_list[:BATCH_SIZE]

    batch_x_image_list = []
    batch_x_label_list = []

    for x_data in batch_data_list:
        image, label = x_data
        image = auto_augment.AutoAugment(image)
        
        batch_x_image_list.append(image)
        batch_x_label_list.append(smooth_one_hot(label, CLASSES))

    batch_x_image_list = np.asarray(batch_x_image_list, dtype = np.float32)
    batch_x_label_list = np.asarray(batch_x_label_list, dtype = np.float32)

    _feed_dict = {
        x_var : batch_x_image_list, 
        x_label_var : batch_x_label_list, 
        is_training : True,
        learning_rate_var : learning_rate
    }
    
    _, loss, x_loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, iter)

    loss_list.append(loss)
    x_loss_list.append(x_loss)
    l2_reg_loss_list.append(l2_reg_loss)
    accuracy_list.append(accuracy)

    if iter % 100 == 0:
        loss = np.mean(loss_list)
        x_loss = np.mean(x_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)
        accuracy = np.mean(accuracy_list)
        train_time = int(time.time() - train_time)

        log_print('[i] iter = {}, loss = {:.4f}, x_loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}, train_time = {}sec'.format(iter, loss, x_loss, l2_reg_loss, accuracy, train_time), log_txt_path)
        
        loss_list = []
        x_loss_list = []
        l2_reg_loss_list = []
        accuracy_list = []
        train_time = time.time()

    if iter % SAVE_ITERATION == 0:
        test_time = time.time()
        test_accuracy_list = []

        for i in range(test_iteration):
            batch_data_list = test_data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            batch_image_data = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype = np.float32)
            batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
            
            for i, (image, label) in enumerate(batch_data_list):
                batch_image_data[i] = image.astype(np.float32)
                batch_label_data[i] = label.astype(np.float32)
            
            _feed_dict = {
                x_var : batch_image_data,
                x_label_var : batch_label_data,
                is_training : False
            }

            accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
            test_accuracy_list.append(accuracy)

        test_time = int(time.time() - test_time)
        test_accuracy = np.mean(test_accuracy_list)

        summary = sess.run(test_accuracy_op, feed_dict = {test_accuracy_var : test_accuracy})
        train_writer.add_summary(summary, iter)

        log_print('[i] iter = {}, test_accuracy = {:.2f}, test_time = {}sec'.format(iter, test_accuracy, test_time), log_txt_path)

saver.save(sess, ckpt_path)
