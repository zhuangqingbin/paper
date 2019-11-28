#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:55:45 2019

@author: jimmy
"""

import keras.backend as K
from keras import activations,regularizers
from keras.engine.topology import Layer, InputSpec
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from keras.optimizers import sgd,adagrad,RMSprop,adam
from keras.models import save_model,load_model
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
def common_model(params,data,train=True,fig=True):
    model = Sequential()
    if params.type == 'FM-R':
        model.add(FMLayer(1, params.k, activation='sigmoid',
                        kernel_regularizer=regularizers.l1_l2(l1=params.l1_reg_rate,l2=params.l2_reg_rate)))
    elif params.type == 'FM':
        model.add(FMLayer(1, params.k, activation='sigmoid'))

    elif params.type == 'FFM-R':
        model.add(FFMLayer(1,data.f,params.k,data.feature2field,activation='sigmoid',
                        kernel_regularizer=regularizers.l1_l2(l1=params.l1_reg_rate,l2=params.l2_reg_rate)))
    elif params.type == 'FFM':
        model.add(FFMLayer(1,data.f,params.k,data.feature2field,activation='sigmoid'))

    elif params.type == 'LR-R':
        model.add(Dense(1,activation='sigmoid',
                        kernel_regularizer=regularizers.l1_l2(l1=params.l1_reg_rate,l2=params.l2_reg_rate)))
    elif params.type == 'LR':
        model.add(Dense(1,activation='sigmoid'))

    else:
        return None
    
    if train:
        if params.optmizer == 'sgd':
            opt = sgd(lr = params.learning_rate)
        elif params.optmizer == 'adagrad':
            opt = adagrad(lr = params.learning_rate)
        elif params.optmizer == 'RMSprop':
            opt = RMSprop(lr = params.learning_rate)
        elif params.optmizer == 'adam':
            opt = adam(lr = params.learning_rate)
        else:
            return 
        
        model.compile(loss = 'binary_crossentropy',
                      optimizer = opt,metrics=['accuracy'])
        history = model.fit(data.tr_X, data.tr_Y,
                  batch_size=params.batch_size,
                  epochs=params.epochs,
                  validation_data=(data.te_X, data.te_Y))
        if fig:
            save_fig(params,history)
            
        
    return model





def save_fig(params,history):
    fig_dir,fig_id = params.fig_dir()
    # 汇总准确率历史数据
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='best')
    plt.savefig(f"{fig_dir}/Acc-{fig_id}.png",dpi=500,bbox_inches ='tight')
    plt.close()
    # 汇总损失函数历史数据
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='best')
    plt.savefig(f"{fig_dir}/Loss-{fig_id}.png",dpi=500,bbox_inches ='tight')
    plt.close()

def get_model(path,type):
    if type == 'LR':
        return load_model(path)
    elif type == 'FM':
        return load_model(path,custom_objects={'FMLayer': FMLayer})
    elif type == 'FFM':
        return load_model(path,custom_objects={'FFMLayer': FFMLayer})
    else:
        return None

class FMLayer(Layer):
    def __init__(self, output_dim,
                 k,activation=None,kernel_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.k = k
        self.activation = activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.w = self.add_weight(name='one', 
                                 shape=(input_dim, self.output_dim),
                                 #initializer='glorot_uniform',
                                 initializer=tf.glorot_uniform_initializer(seed=1994),
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)
        self.v = self.add_weight(name='two', 
                                 shape=(input_dim, self.k),
                                 #initializer='glorot_uniform',
                                 initializer=tf.glorot_uniform_initializer(seed=1994),
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)
        self.b = self.add_weight(name='bias', 
                                 shape=(self.output_dim,),
                                 #initializer='zeros',
                                 initializer=tf.glorot_uniform_initializer(seed=1994),
                                 trainable=True)

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        X_square = K.square(inputs)

        xv = K.square(K.dot(inputs, self.v))
        xw = K.dot(inputs, self.w)

        p = 0.5 * K.sum(xv - K.dot(X_square, K.square(self.v)), 1)
        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))
        
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim




class FFMLayer(Layer):
    def __init__(self, output_dim,
                 f,k,feature2field,
                 activation=None,kernel_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FFMLayer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.f = f
        self.k = k
        self.feature2field = feature2field
        self.activation = activations.get(activation)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.p = input_shape[1]
        
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.w = self.add_weight(name='one', 
                                 shape=(input_dim, self.output_dim),
                                 initializer=tf.glorot_uniform_initializer(seed=1994),
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)
        self.v = self.add_weight(name='two', 
                                 shape=(input_dim,self.f,self.k),
                                 initializer=tf.glorot_uniform_initializer(seed=1994),
                                 regularizer=self.kernel_regularizer,
                                 trainable=True)
        self.b = self.add_weight(name='bias', 
                                 shape=(self.output_dim,),
                                 initializer=tf.glorot_uniform_initializer(seed=1994),
                                 trainable=True)

        super(FFMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        
        print(inputs[:,1])
        xw = K.dot(inputs, self.w)
        
        rp = K.constant(0, dtype='float32')
        for i in range(self.p):
            for j in range(i+1,self.p):
                vifj = self.v[i, self.feature2field[j]]
                vjfi = self.v[j, self.feature2field[i]]
                vivj = K.sum(vifj*vjfi)
                xixj = inputs[:,i]*inputs[:,j]
                rp += vivj*xixj
#                vivj = K.sum(multiply([vifj,vjfi]))
#                xixj = multiply([inputs[:,i],inputs[:,j]])
#                rp += multiply([vivj,xixj])
        rp = K.reshape(rp, (-1, self.output_dim))
        
        
        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))
        
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim




