"""
Copyright Netherlands eScience Center
Function     : Emotion recognition and forecast with BBConvLSTM
Author       : Yang Liu & Tianyi Zhang
Contributor  : Tianyi Zhang (Centrum Wiskunde & Informatica)
First Built  : 2021.02.24
Last Update  : 2021.03.01
Library      : Pytorth, Numpy, os, NEmo, matplotlib
Description  : This script serves to test the prediction skill of deep neural networks in emotion recognition and forecast. The Bayesian convolutional Long Short Time Memory neural network with Bernoulli approximate variational inference is used to deal with this spatial-temporal sequence problem. We use Pytorch as the deep learning framework.

Many to one prediction.

Return Values: pkl model and figures

This project is a joint venture between NLeSC and CWI

Reference:
Gal, Y., & Ghahramani, Z. (2016, June). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning (pp. 1050-1059).
Xingjian, S. H. I., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems (pp. 802-810).

"""

import sys
import numbers
import pickle
# for data loading
import os
# for pre-processing and machine learning
import numpy as np
#import sklearn
from scipy.signal import resample
import torch
import torch.nn.functional

sys.path.append("../")
import nemo
import nemo.ConvLSTM
import nemo.BBConvLSTM
import nemo.function
import nemo.metric

# for visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

##################################################################
#########                 functions                       ########
##################################################################

def MSE(x, y):
    """
    Calculate the MSE. x is input series and y is reference series.
    """
    mse = np.mean((x-y)**2)
        
    return mse

def load_data(window_size, num_s, v_a):
    """
    For long cuts of data and the type is pickle.
    """
    f = open('../Data_CASE/data_{}s'.format(int(window_size/100)),'rb')
    data = pickle.load(f)
    f.close()
    
    samples = data["Samples"]
    labels = data["label_s"]
    subject_id = data["Subject_id"]
    
    x_train = samples[np.where(subject_id!=num_s)[0],:,0:4]
    x_test = samples[np.where(subject_id==num_s)[0],:,0:4]
    y_train = np.zeros([0,int(window_size/seq),1])
    y_test = np.zeros([0,int(window_size/seq),1])
    for i in range(len(labels)):
        sig = resample(labels[i][:,v_a],int(window_size/seq)).reshape([1,-1,1])/9
        if subject_id[i] == num_s:
            y_test = np.concatenate([y_test,sig],axis = 0)
        else:
            y_train = np.concatenate([y_train,sig],axis = 0)

    return x_train, x_test, y_train, y_test

def folding(data, dim_x, dim_y):
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

def train_model(window_size, dim_x, dim_y, 
                hidden_channels, kernel_size, num_s, v_a):
    model_type = "BBConvLSTM"
    x_train, x_test, y_train, y_test = load_data(window_size, num_s, v_a)
    
    if not os.path.exists("../baseline_result/%s/%s/"%(model_type,v_a)):
        os.makedirs("../baseline_result/%s/%s/"%(model_type,v_a))     
    if not os.path.exists("../model/%s/%s/"%(model_type,v_a)):
        os.makedirs("../model/%s/%s/"%(model_type,v_a))

    # choose the target dimension for reshaping of the signals
    batch_train_size, sample_x_size, channels = x_train.shape
    batch_test_size, sample_x_size, _ = x_test.shape
    _, sample_y_size, _ = y_train.shape
    series_x_len = sample_x_size // (dim_y * dim_x)
    series_y_len = sample_y_size // (dim_y * dim_x)
    
    # reshape the input and labels
    sample_train_xy = folding(x_train, dim_x, dim_y)
    sample_test_xy = folding(x_test, dim_x, dim_y)
    label_c_train_xy = folding(y_train, dim_x, dim_y)
    label_c_test_xy = folding(y_test, dim_x, dim_y)
    
    # length diff for training (due to the uneven length of training data and validing data)
    len_diff = series_x_len / series_y_len
    # for the output at certain loop iteration
    diff_output_iter = np.arange(len_diff-1, series_x_len, len_diff)
        
    print ('*******************  configuration before training  *********************')
    # specifications of neural network
    batch_train_size, sample_x_size, channels = x_train.shape
    # here we input a sequence and predict the next step only
    learning_rate = 0.001
    num_epochs = 50
    # probability of dropout
    p = 0.5 # 0.5 for Bernoulli (binary) distribution
    print (torch.__version__)
    # check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print("Is CUDA available? {}".format(use_cuda))
    # CUDA settings torch.__version__ must > 0.4
    # !!! This is important for the model!!! The first option is gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ('*******************  cross validation and testing data  *********************')
    # check the sequence length
    _, sequence_len, height, width, _ = sample_train_xy.shape
    # mini-batch
    mini_batch_size = 28
    # iterations
    iterations = batch_train_size // mini_batch_size
    if batch_train_size % mini_batch_size != 0:
        extra_loop = True
        iterations += 1
    else:
        extra_loop = False
    
    print ('*******************  run LSTM  *********************')
    print ('The model is designed to make many to one prediction.')
    print ('A series of multi-chanel variables will be input to the model.')
    print ('The model learns by verifying the output at each timestep.')
    # initialize our model
    model = nemo.BBConvLSTM.BBConvLSTM(input_channels, hidden_channels, kernel_size, p).to(device)
    loss_fn = torch.nn.MSELoss(reduction="mean")
    # stochastic gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # Adam optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #print(model)
    #print(loss_fn)
    #print(optimiser)        
    
    print('##############################################################')
    print('##################  start training loop  #####################')
    print('##############################################################')
    hist = np.zeros(num_epochs * iterations)
    # loop of epoch
    for t in range(num_epochs):
        for i in range(iterations):
            # Clear stored gradient
            model.zero_grad()
            # loop
            loop_num = mini_batch_size
            if i == iterations - 1:
                if extra_loop:
                    loop_num = batch_train_size % mini_batch_size
            for j in range(loop_num):
                # loop of timestep
                # critical step for verifying the loss
                crit_step = 0
                for timestep in range(sequence_len):
                    # hidden state re-initialized inside the model when timestep=0
                    #################################################################################
                    ########          create input tensor with multi-input dimension         ########
                    #################################################################################
                    # create variables
                    x_input = np.stack((sample_train_xy[i*mini_batch_size+j,timestep,:,:,0],
                                        sample_train_xy[i*mini_batch_size+j,timestep,:,:,1],
                                        sample_train_xy[i*mini_batch_size+j,timestep,:,:,2],
                                        sample_train_xy[i*mini_batch_size+j,timestep,:,:,3])) #vstack,hstack,dstack
                    x_var = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,dim_y,dim_x)).to(device)
                    # Forward pass
                    y_pred, _ = model(x_var, timestep)
                    # counter for computing the loss
                    if timestep in diff_output_iter: # determined by the length of label                    
                        #################################################################################
                        ########       create training tensor with multi-input dimension         ########
                        #################################################################################
                        y_train_stack = np.stack((label_c_train_xy[i*mini_batch_size+j,crit_step,:,:])) #vstack,hstack,dstack
                        y_var = torch.autograd.Variable(torch.Tensor(y_train_stack).view(-1,hidden_channels[-1],height,width)).to(device)
                        crit_step = crit_step + 1 # move forward
                        #################################################################################   
                        # choose training data
                        y_train = y_var    
                        # torch.nn.functional.mse_loss(y_pred, y_train) can work with (scalar,vector) & (vector,vector)
                        # Please Make Sure y_pred & y_train have the same dimension
                        # accumulate loss
                        if timestep == diff_output_iter[0]:
                            loss = loss_fn(y_pred, y_train)
                        else:
                            loss += loss_fn(y_pred, y_train)
            # print loss at certain iteration
            if i % 10 == 0:
                print("Epoch {} Iteration {} MSE: {:0.3f}".format(t, i, loss.item()))
                # Gradcheck requires double precision numbers to run
                #res = torch.autograd.gradcheck(loss_fn, (y_pred.double(), y_train.double()), eps=1e-6, raise_exception=True)
                #print(res)
            hist[i+t*iterations] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()
    
            # Backward pass
            loss.backward(retain_graph=True)

            # Update parameters
            optimiser.step()
            
    # save the model
    # (recommended) save the model parameters only
    torch.save(model.state_dict(), os.path.join('../model/%s/%s/'%(model_type,v_a),
                                                'emotion_%s_%s.pkl'%(model_type,v_a)))
    ########### attention !! no early stopping is implemented here!! ################
    
    #################################################################################
    ###########                 after training statistics                 ###########
    #################################################################################
    print ("*******************  Loss with time  **********************")
    fig00 = plt.figure()
    plt.plot(hist, 'r', label="Training loss")
    plt.xlabel('Iterations')
    plt.ylabel('MSE Error')
    plt.legend()
    fig00.savefig(os.path.join('../model/%s/%s/'%(model_type,v_a),
                               'train_mse_error_%s_%s.png'%(model_type,v_a)),dpi=150)
    
    print ("*******************  Loss with time (log)  **********************")
    fig01 = plt.figure()
    plt.plot(np.log(hist), 'r', label="Training loss")
    plt.xlabel('Iterations')
    plt.ylabel('Log mse error')
    plt.legend()
    plt.show()
    fig01.savefig(os.path.join('../model/%s/%s/'%(model_type,v_a),
                               'train_log_mse_error_%s_%s.png'%(model_type,v_a)),dpi=150)
    
    #################################################################################
    ########                           prediction                            ########
    #################################################################################
    print('##############################################################')
    print('###################  start prediction loop ###################')
    print('##############################################################')
    # forecast array
    pred_label = np.zeros((batch_test_size, series_y_len, dim_y, dim_x),dtype=float)
    # calculate loss for each sample
    hist_label = np.zeros(batch_test_size)
    for n in range(batch_test_size):
        # Clear stored gradient
        model.zero_grad()
        # critical step for saving the output
        crit_step_pred = 0        
        for timestep in range(sequence_len):
            x_input = np.stack((sample_test_xy[n,timestep,:,:,0],
                                sample_test_xy[n,timestep,:,:,1],
                                sample_test_xy[n,timestep,:,:,2],
                                sample_test_xy[n,timestep,:,:,3]))
            x_var_pred = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,dim_y,dim_x),
                                                 requires_grad=False).to(device)
            # make prediction
            last_pred, _ = model(x_var_pred, timestep)
            # counter for prediction
            if timestep in diff_output_iter: # determined by the length of label
                # GPU data should be transferred to CPU
                pred_label[n,crit_step_pred,:,:] = last_pred[0,0,:,:].cpu().data.numpy()
                crit_step_pred = crit_step_pred + 1 # move forward
        # compute the error for each sample
        hist_label[n] = MSE(label_c_test_xy[n,:,:,:,0], pred_label[n,:,:,:])
    
    # save prediction as npz file
    np.savez("../baseline_result/%s/%s/predict_%s_%ss_%s.npz"%(model_type,v_a,model_type,int(window_size/100),num_s),
             t1=pred_label, label = label_c_test_xy)
    # plot the error
    print ("*******************  Loss with time  **********************")
    fig02 = plt.figure()
    plt.plot(hist_label, 'r', label="Testing loss")
    plt.xlabel('Sample')
    plt.ylabel('MSE Error')
    plt.legend()
    fig02.savefig(os.path.join("../baseline_result/%s/%s"%(model_type,v_a),
                               'pred_mse_error_%s_%s.png'%(model_type,v_a)),dpi=150)
    
    #####################################################################################
    ########         visualization of prediction and implement metrics           ########
    #####################################################################################
    # compute mse
    mse = MSE(label_c_test_xy[:,:,:,:,0], pred_label)
    print(mse)
    print("=====Finished %s======= mse = %.4f====="%(num_s,mse))
    
    return mse


def losocv(window_size, dim_x, dim_y, hidden_channels, kernel_size):
    # leave one subject out cross validation
    model_type = "BBConvLSTM"
    print("=====Start training %s, window = %ss, seg = %ss====="%(model_type,int(window_size/100),int(dim_x)))
    mse_v = []
    mse_a = []
    for i in np.arange(1,31):
        mse_v.append(train_model(window_size, dim_x, dim_y,
                                 hidden_channels, kernel_size, i, 0))
        mse_a.append(train_model(window_size, dim_x, dim_y,
                                 hidden_channels, kernel_size, i, 1))
    if not os.path.exists("../baseline_result/%s/"%model_type):
        os.makedirs("../baseline_result/%s/"%model_type) 
    np.savez("../baseline_result/%s/results_%s.npz"%(model_type,model_type),
             mse_v = mse_v,
             mse_a = mse_a,
             mse = [np.mean(mse_v),np.mean(mse_a)])
    print("=====Finished training %s, window = %ss, seg = %ss=========="%(model_type,int(window_size/100),int(dim_x)))
    print("====== mse_v = %.4f, mse_a = %.4f ====="%(np.mean(mse_v),np.mean(mse_a)))
    
if __name__=="__main__":
    ################################################################################# 
    #########                 parameters and variables                       ########
    #################################################################################
    # please specify the constants for input data
    window_size = 2000 # down-sampling constant
    seq = 20
    #v_a = 0 # valance = 0, arousal = 1
    # leave-one-out training and testing
    #num_s = 2
    #-------------------------------------------
    # please specify the hyperparameters for network
    dim_x = 5
    dim_y = 5
    input_channels = 4
    hidden_channels = [3, 2, 1] # number of channels & hidden layers, the channels of last layer is the channels of output, too
    kernel_size = 1
    #################################################################################
    # test module
    print("run 1st test")
    train_model(window_size, dim_x, dim_y, hidden_channels, kernel_size, 1, 0)
    print("1st test complete")
    # test module
    print("run 2nd test")
    losocv(window_size, dim_x, dim_y, hidden_channels, kernel_size)
    print("2nd test complete")    