# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,LSTM, Conv1D,Bidirectional
from tensorflow.keras.activations import sigmoid
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import mean_squared_error
import os

def init_tf_gpus():
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Physical GPUs available: %d  -- Logical GPUs available: %d" % (len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            print(e) 

def load_data(window_size,num_s,v_a):
    data = np.load("./dataset/CASE/data_%ss.npz"%int(window_size/100))
    samples = data["Samples"]
    labels = data["Labels_c"]
    subject_id = data["Subject_id"]
    
    x_train = samples[np.where(subject_id!=num_s)[0],:,0:5]
    x_test = samples[np.where(subject_id==num_s)[0],:,0:5]
    
    y_train = labels[np.where(subject_id!=num_s)[0],:,v_a]/10
    y_test = labels[np.where(subject_id==num_s)[0],:,v_a]/10
    return x_train, x_test, y_train, y_test

def create_mode(model_type,window_size):
    input_signals = Input(shape=(window_size,5))
    if(model_type == "LINEAR"):        
        x = Dense(1,activation = sigmoid)(input_signals)
    elif (model_type == "PLOY"):
        x = Dense(30,activation = sigmoid)(input_signals)
        x = Dense(5,activation = sigmoid)(x)
        x = Dense(1,activation = sigmoid)(x)
    elif (model_type == "LSTM"):
        x = LSTM(window_size,recurrent_dropout = 0.2,kernel_initializer=initializers.he_uniform())(input_signals)
    elif (model_type == "BiLSTM"):
        x = Bidirectional(LSTM(int(window_size/2),recurrent_dropout = 0.2,kernel_initializer=initializers.he_uniform()))(input_signals)
    elif (model_type == "CNNLSTM"):
        x = Conv1D(4, 256, activation='relu',input_shape=(window_size,5),padding = "same")(input_signals)
        x = Conv1D(8, 128, activation='relu',padding = "same")(x)
        x = Conv1D(16, 64, activation='relu',padding = "same")(x)
        x = Conv1D(64, 32, activation='relu',padding = "same")(x)
        x = Conv1D(128, 16, activation='relu',padding = "same")(x)
        x = LSTM(window_size,recurrent_dropout = 0.2,kernel_initializer=initializers.he_uniform())(x)
    else:
        print("Not a supported model")
        exit(0) 
    model = Model(input_signals, x)
    return model

def train_model(window_size,model_type,num_s,v_a):
    init_tf_gpus()
    x_train, x_test, y_train, y_test = load_data(window_size,num_s,v_a)
    model = create_mode(model_type,window_size)
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

def losocv(window_size,model_type):
    print("=====Start training %s, window = %ss========="%(model_type,int(window_size/100)))
    mse_v = []
    mse_a = []
    for i in range(30):
        mse_v.append(train_model(window_size,model_type,i,0))
        mse_a.append(train_model(window_size,model_type,i,1))
    if not os.path.exists("./baseline_result/%s/"%model_type):
        os.makedirs("./baseline_result/%s/"%model_type) 
    np.savez("./baseline_result/%s/results_%s.npz"%(model_type,model_type), 
             mse_v= mse_v, 
             mse_a = mse_a, 
             mse = [np.mean(mse_v),np.mean(mse_a)])
    print("=====Finished training %s, window = %ss====="%(model_type,int(window_size/100)))
    print("====== mse_v = %.4f, mse_a = %.4f ====="%(np.mean(mse_v),np.mean(mse_a)))