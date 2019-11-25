#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 20:08:24 2019

@author: jimmy
"""

import tensorflow as tf
import os
class FFM(object):
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
        self.optmizer = params.optmizer
        
        dir_path = '%s/models/' % os.getcwd()
        data_path = f'N{params.n}_{params.noise_len}_{params.numeric_len}_{params.object_len}_{str(params.object_nums)}/'
        name = f'O{params.optmizer}_E{params.epoch}_K{params.k}_Lr{params.learning_rate}_B{params.batch_size}_L1{params.l1_reg_rate}_L2{params.l2_reg_rate}'
        self.checkpoint_dir = dir_path+data_path+name+'_FFM'
        
        

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
            #print('self.linear_terms:')
            #print(self.linear_terms)

        with tf.variable_scope('nolinear_layer'):
            self.v = tf.get_variable('v', shape=[self.p, self.f, self.k], dtype='float32',
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # v:pxfxk
            self.field_cross_interaction = tf.constant(0, dtype='float32')
            # 每个特征
            for i in range(self.p):
                # 寻找没有match过的特征，也就是论文中的j = i+1开始
                for j in range(i + 1, self.p):
                    #print('i:%s,j:%s' % (i, j))
                    # vifj
                    vifj = self.v[i, self.feature2field[j]]
                    # vjfi
                    vjfi = self.v[j, self.feature2field[i]]
                    # vi · vj
                    vivj = tf.reduce_sum(tf.multiply(vifj, vjfi))
                    # xi · xj
                    xixj = tf.multiply(self.X[:, i], self.X[:, j])
                    self.field_cross_interaction += tf.multiply(vivj, xixj)
            self.field_cross_interaction = tf.reshape(self.field_cross_interaction, (self.batch_size, 1))
            #print('self.field_cross_interaction:')
            #print(self.field_cross_interaction)
        
        self.y_out_tmp = tf.nn.sigmoid(tf.add(self.linear_terms, self.field_cross_interaction))
        #print('y_out_prob:')
        #print(self.y_out)
        
        # Loss function
        self.y_out = tf.clip_by_value(self.y_out_tmp,1e-8,1-1e-8)
        log_loss = self.y*tf.log(self.y_out)+(1-self.y)*tf.log(1-self.y_out)
        self.loss = tf.reduce_mean(-log_loss)
        ## regularation
        self.loss += tf.contrib.layers.l1_regularizer(self.l1_reg_rate)(self.w1)
        self.loss += tf.contrib.layers.l1_regularizer(self.l1_reg_rate)(self.v)
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.w1)
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.v)
        
        
        self.global_step = tf.Variable(0, trainable=False)
        if self.optmizer == 'GD':
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optmizer == 'Adagrad':
            opt = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optmizer == 'Adadelta':
            opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif self.optmizer == 'Adam':
            opt = tf.train.AdamOptimizer(self.learning_rate)
        
        trainable_params = tf.trainable_variables()
        #print(trainable_params)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)
        
        
        
        
    def train(self, sess, x, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.X: x,
            self.y: label
        })
#        _ , loss = sess.run([self.train_step,self.loss], feed_dict={
#            self.X: x,
#            self.y: label
#        })
        return loss

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

    def save(self,sess):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        saver = tf.train.Saver()
        saver.save(sess, save_path=f'{self.checkpoint_dir}/saver')

    def restore(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=f'{self.checkpoint_dir}/saver')
