# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,ConvLSTM2D
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import mean_squared_error
from tensorflow.compat.v1.keras.backend import set_session


def init_tf_gpus():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = "0" #only the gpu 0 is allowed
    set_session(tf.compat.v1.Session(config=config))


def folding(data, dim_x,dim_y):
    if(data.ndim ==3):
        batch_size, sample_size, channels = data.shape
    elif (data.ndim ==2):
        batch_size, sample_size = data.shape
        channels =1
    else:
        print("Data format not supported!")
        exit(0)
    series_len = sample_size // (dim_x * dim_y)
    return np.reshape(data, [batch_size, series_len, dim_y, dim_x, channels])
    
    
def load_data(window_size,dim_x,dim_y,num_s,v_a):
    data = np.load("./dataset/CASE/data_%ss.npz"%int(window_size/100))
    samples = data["Samples"]
    labels = data["Labels_c"]
    subject_id = data["Subject_id"]
    
    x_train = folding(samples[np.where(subject_id!=num_s)[0],:,0:5],dim_x,dim_y)
    x_test = folding(samples[np.where(subject_id==num_s)[0],:,0:5],dim_x,dim_y)
    
    y_train = folding(labels[np.where(subject_id!=num_s)[0],:,v_a]/10,dim_x,dim_y)
    y_test = folding(labels[np.where(subject_id==num_s)[0],:,v_a]/10,dim_x,dim_y)
    return x_train, x_test, y_train, y_test

def create_mode(window_size, dim_x, dim_y, hidden_channels, kernel_size):
    input_signals = Input(shape=(window_size//(dim_x * dim_y),dim_x,dim_y,5))
    x = ConvLSTM2D(filters=hidden_channels, kernel_size=(kernel_size, kernel_size),
                   padding='same', return_sequences=True)(input_signals)
    x = ConvLSTM2D(filters=3, kernel_size=(kernel_size, kernel_size),
                   padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=1, kernel_size=(kernel_size, kernel_size),
                   padding='same', return_sequences=True)(x)
    model = Model(input_signals, x) 
    return model


def train_model(window_size,dim_x,dim_y,hidden_channels, kernel_size,num_s,v_a):
    model_type = "CONVLSTM"
    x_train, x_test, y_train, y_test = load_data(window_size,dim_x,dim_y,num_s,v_a)
    model = create_mode(window_size,dim_x, dim_y,hidden_channels, kernel_size)
    if not os.path.exists("./baseline_result/%s/%s/"%(model_type,v_a)):
        os.makedirs("./baseline_result/%s/%s/"%(model_type,v_a))     
    if not os.path.exists("./model/%s/%s/"%(model_type,v_a)):
        os.makedirs("./model/%s/%s/"%(model_type,v_a))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ModelCheckpoint('./model/%s/%s/model_%s.h5'%(model_type,v_a,num_s),monitor='val_loss', save_best_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', min_lr=0)
    ]
    RMSprop= optimizers.RMSprop(lr=0.001)
    model.compile(loss=tf.keras.losses.LogCosh(),
                          optimizer=RMSprop,
                          metrics=["mse"])
    history = model.fit(x_train, y_train, 
                        batch_size=128, 
                        callbacks=callbacks, 
                        epochs=1000,
                        verbose=0, 
                        validation_split=0.2)
    t1 = model.predict(x_test,batch_size=32)
    mse = mean_squared_error(y_test.reshape([-1,1]), t1.reshape([-1,1]))    
    if not os.path.exists("./baseline_result/%s/%s/"%(model_type,v_a)):
        os.makedirs("./baseline_result/%s/%s/"%(model_type,v_a))  
    np.savez("./baseline_result/%s/%s/predict_%ss_%s.npz"%(model_type,v_a,int(window_size/100),num_s),predict = t1, label = y_test)
    print("=====Finished %s======= mse = %.4f====="%(num_s,mse))
    return mse    
    
def losocv(window_size,dim_x,dim_y,hidden_channels, kernel_size):
    init_tf_gpus()
    model_type = "CONVLSTM"
    print("=====Start training %s, window = %ss, seg = %ss====="%(model_type,int(window_size/100),int(dim_x)))
    mse_v = []
    mse_a = []
    for i in range(30):
        mse_v.append(train_model(window_size,dim_x,dim_y,hidden_channels, kernel_size,i,0))
        mse_a.append(train_model(window_size,dim_x,dim_y,hidden_channels, kernel_size,i,1))
    if not os.path.exists("./baseline_result/%s/"%model_type):
        os.makedirs("./baseline_result/%s/"%model_type) 
    np.savez("./baseline_result/%s/results_%s.npz"%(model_type,model_type), 
             mse_v= mse_v, 
             mse_a = mse_a, 
             mse = [np.mean(mse_v),np.mean(mse_a)])
    print("=====Finished training %s, window = %ss, seg = %ss=========="%(model_type,int(window_size/100),int(dim_x)))
    print("====== mse_v = %.4f, mse_a = %.4f ====="%(np.mean(mse_v),np.mean(mse_a)))    
