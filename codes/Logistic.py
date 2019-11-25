#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 14:31:14 2019

@author: jimmy
"""

import tensorflow as tf
import os
class Logistic(object):
    def __init__(self,params,data):
        self.f = data.f
        self.p = data.p
        self.k = params.k
        self.feature2field = data.feature2field
        self.tr_Y,self.te_Y = data.tr_Y,data.te_Y
        self.tr_X,self.te_X = data.tr_X,data.te_X
          
        self.learning_rate = params.learning_rate
        self.batch_size = params.batch_size
        self.l1_reg_rate = params.l1_reg_rate
        self.l2_reg_rate = params.l2_reg_rate
        self.epoch = params.epoch
        self.name = 'Logistic'

    def build_model(self):
        self.X = tf.placeholder('float32', [self.batch_size, self.p])
        self.y = tf.placeholder('float32', [None, 1])

        # linear part
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[1],
                                initializer=tf.zeros_initializer())
            self.w1 = tf.get_variable('w1', shape=[self.p, 1],
                                      initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.linear_terms = tf.add(tf.matmul(self.X, self.w1), b)
            print('self.linear_terms:')
            print(self.linear_terms)

        with tf.variable_scope('nolinear_layer'):
            self.w2 = tf.get_variable('w2', shape=[self.p*(self.p-1)//2, 1], dtype='float32',
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # v:pxfxk
            self.field_cross_interaction = tf.constant(0, dtype='float32')
            # 每个特征
            init_index = 0
            for i in range(self.p):
                # 寻找没有match过的特征，也就是论文中的j = i+1开始
                for j in range(i + 1, self.p):
                    print('i:%s,j:%s' % (i, j))
                    # vifj
                    # vifj = self.v[i, self.feature2field[j]]
                    # vjfi
                    # vjfi = self.v[j, self.feature2field[i]]
                    # vi · vj
                    # vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))
                    # wij
                    wij = self.w2[init_index]
                    init_index += 1
                    # xi · xj
                    xixj = tf.multiply(self.X[:, i], self.X[:, j])
                    self.field_cross_interaction += tf.multiply(wij, xixj)
            self.field_cross_interaction = tf.reshape(self.field_cross_interaction, (self.batch_size, 1))
            print('self.field_cross_interaction:')
            print(self.field_cross_interaction)
        
        self.y_out = tf.nn.sigmoid(tf.add(self.linear_terms, self.field_cross_interaction))
        print('y_out_prob:')
        print(self.y_out)
        
        # Loss function
        log_loss = self.y*tf.log(self.y_out)+(1-self.y)*tf.log(1-self.y_out)
        self.loss = tf.reduce_mean(-log_loss)
        ## regularation
        self.loss += tf.contrib.layers.l1_regularizer(self.l1_reg_rate)(self.w1)
        self.loss += tf.contrib.layers.l1_regularizer(self.l1_reg_rate)(self.w2)
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.w1)
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.w2)
        
        
        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        print(trainable_params)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, x, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.X: x,
            self.y: label
        })
        return loss, step

    def cal(self, sess, x, label):
        y_out_prob_ = sess.run([self.y_out], feed_dict={
            self.X: x,
            self.y: label
        })
        return y_out_prob_, label

    def predict(self, sess, x):
        result = sess.run([self.y_out], feed_dict={
            self.X: x
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        if os.path.exists(f'{path}_{self.name}'):
            saver.save(sess, save_path=f'{path}_{self.name}/saver')
        else:
            os.mkdir(f'{path}_{self.name}')
            saver.save(sess, save_path=f'{path}_{self.name}/saver')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=f'{path}_{self.name}/saver')
