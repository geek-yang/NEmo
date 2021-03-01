"""
# Copyright Netherlands eScience Center and Centrum Wiskunde & Informatica <br>
** Function     : Emotion recognition and forecast with BBConvLSTM** <br>
** Author       : Yang Liu** <br>
** Contributor  : Tianyi Zhang (Centrum Wiskunde & Informatica)<br>
** First Built  : 2021.02.04 ** <br>
** Last Update  : 2021.02.04 ** <br>
** Library      : Pytorth, Numpy, os, DLACs, matplotlib **<br>
Description     : This notebook serves to test the prediction skill of deep neural networks in emotion recognition and forecast. The Bayesian convolutional Long Short Time Memory neural network with Bernoulli approximate variational inference is used to deal with this spatial-temporal sequence problem. We use Pytorch as the deep learning framework. <br>
<br>
** Many to one prediction.** <br>

Return Values   : Time series and figures <br>

**This project is a joint venture between NLeSC and CWI** <br>

The method comes from the study by Shi et. al. (2015) Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. <br>
"""

import sys
import numbers

# for data loading
import os
# for pre-processing and machine learning
import numpy as np
import csv
#import sklearn
#import scipy
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

################################################################################# 
#########                           datapath                             ########
#################################################################################
# please specify data path
datapath = 'C:\\Users\\nosta\\NEmo\\Data_CASE'
output_path = 'C:\\Users\\nosta\\NEmo\\results'
model_path = 'C:\\Users\\nosta\\NEmo\\models'
#################################################################################

if __name__=="__main__":
    print ('*********************** extract variables *************************')
    data = np.load(os.path.join(datapath, "data_10s.npz"))
    #data = np.load(os.path.join(datapath, "data_2s.npz"))
    #data = np.load(os.path.join(datapath, "data_0.5s.npz"))
    #################################################################################
    #########                        data gallery                           #########
    #################################################################################
    sample = data["Samples"][:] # (batch_size, sample_size, channels)
    label_c = data["Labels_c"][:] # (batch_size, sample_size, 2)
    label = data["Labels"][:] # (batch_size, 2)
    subject = data["Subject_id"][:] # (batch_size, 2)
    video_label = data["Video_labels"][:] # (batch_size,1)
    
    # leave-one-out training and testing
    num_s = 2
    sample_train = sample[np.where(subject!=num_s)[0],:,0:5]
    sample_test = sample[np.where(subject==num_s)[0],:,0:5]
    
    label_c_train = label_c[np.where(subject!=num_s)[0],:,:] / 10 # normalize
    label_c_test = label_c[np.where(subject==num_s)[0],:,:] / 10 # normalize
    
    #################################################################################
    #########                      pre-processing                           #########
    #################################################################################
    # choose the target dimension for reshaping of the signals
    batch_train_size, sample_size, channels = sample_train.shape
    batch_test_size, _, _ = sample_test.shape
    _, _, label_channels = label_c_train.shape
    x_dim = 5
    y_dim = 5
    series_len = sample_size // (y_dim * x_dim)
    # reshape the input and labels
    sample_train_xy = np.reshape(sample_train,[batch_train_size, series_len, y_dim, x_dim, channels])
    sample_test_xy = np.reshape(sample_test,[batch_test_size, series_len, y_dim, x_dim, channels])
    label_c_train_xy = np.reshape(label_c_train,[batch_train_size, series_len, y_dim, x_dim, label_channels])
    label_c_test_xy = np.reshape(label_c_test,[batch_test_size, series_len, y_dim, x_dim, label_channels])
    #################################################################################
    #########                       normalization                           #########
    #################################################################################
    print('================  extract individual variables  =================')
    sample_1 = sample_train_xy[:,:,:,:,0]
    sample_2 = sample_train_xy[:,:,:,:,1]
    sample_3 = sample_train_xy[:,:,:,:,2]
    sample_4 = sample_train_xy[:,:,:,:,3]
    sample_5 = sample_train_xy[:,:,:,:,4]
    
    sample_1_test = sample_test_xy[:,:,:,:,0]
    sample_2_test = sample_test_xy[:,:,:,:,1]
    sample_3_test = sample_test_xy[:,:,:,:,2]
    sample_4_test = sample_test_xy[:,:,:,:,3]
    sample_5_test = sample_test_xy[:,:,:,:,4]   
    
    label_c_valance = label_c_train_xy[:,:,:,:,0]
    label_c_arousal = label_c_train_xy[:,:,:,:,1]
    
    label_c_test_valance = label_c_test_xy[:,:,:,:,0]
    label_c_test_arousal = label_c_test_xy[:,:,:,:,1]
    
    # using indicator for training
    # video_label_3D = np.repeat(video_label[:,np.newaxis,:],series_len,1)
    # video_label_4D = np.repeat(video_label_3D[:,:,np.newaxis,:],y_dim,2)
    # video_label_xy = np.repeat(video_label_4D[:,:,:,np.newaxis,:],x_dim,3)
    # video_label_xy.astype(float)
    
    print ('*******************  create basic dimensions for tensor and network  *********************')
    # specifications of neural network
    input_channels = 5
    hidden_channels = [4, 3, 2] # number of channels & hidden layers, the channels of last layer is the channels of output, too
    #hidden_channels = [3, 3, 3, 3, 2]
    #hidden_channels = [2]
    kernel_size = 3
    # here we input a sequence and predict the next step only
    learning_rate = 0.01
    num_epochs = 200
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
    # mini-batch
    mini_batch_size = 64
    # iterations
    iterations = batch_train_size // mini_batch_size
    if batch_train_size % mini_batch_size != 0:
        extra_loop = "True"
        iterations += 1
        
    print ('*******************  run BBConvLSTM  *********************')
    print ('The model is designed to make many to one prediction.')
    print ('A series of multi-chanel variables will be input to the model.')
    print ('The model learns by verifying the output at each timestep.')
    # check the sequence length
    _, sequence_len, height, width = sample_1.shape
    # initialize our model
    model = nemo.BBConvLSTM.BBConvLSTM(input_channels, hidden_channels, kernel_size, p).to(device)
    loss_fn = torch.nn.MSELoss(size_average=True)
    # stochastic gradient descent
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # Adam optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    print(loss_fn)
    print(optimiser)

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
                for timestep in range(sequence_len):
                    # hidden state re-initialized inside the model when timestep=0
                    #################################################################################
                    ########          create input tensor with multi-input dimension         ########
                    #################################################################################
                    # create variables
                    x_input = np.stack((sample_1[i*mini_batch_size+j,timestep,:,:],
                                        sample_2[i*mini_batch_size+j,timestep,:,:],
                                        sample_3[i*mini_batch_size+j,timestep,:,:],
                                        sample_4[i*mini_batch_size+j,timestep,:,:],
                                        sample_5[i*mini_batch_size+j,timestep,:,:])) #vstack,hstack,dstack
                    x_var = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,height,width)).to(device)
                    #################################################################################
                    ########       create training tensor with multi-input dimension         ########
                    #################################################################################
                    y_train_stack = np.stack((label_c_valance[i*mini_batch_size+j,timestep,:,:],
                                              label_c_arousal[i*mini_batch_size+j,timestep,:,:])) #vstack,hstack,dstack
                    y_var = torch.autograd.Variable(torch.Tensor(y_train_stack).view(-1,hidden_channels[-1],height,width)).to(device)
                    #################################################################################   
                    # Forward pass
                    y_pred, _ = model(x_var, timestep)
                    # choose training data
                    y_train = y_var        
                    # torch.nn.functional.mse_loss(y_pred, y_train) can work with (scalar,vector) & (vector,vector)
                    # Please Make Sure y_pred & y_train have the same dimension
                    # accumulate loss
                    if timestep == 0:
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
            loss.backward()

            # Update parameters
            optimiser.step()
        
        # save the model
        if t % 50 == 0:
        # (recommended) save the model parameters only
        torch.save(model.state_dict(), os.path.join(model_path,'BBconvlstm_emotion_epoch_{}.pkl'.format(t)))
        # save the entire model
        #torch.save(model, os.path.join(output_path,'convlstm.pkl'))
        
    # save the model    
    torch.save(model.state_dict(), os.path.join(model_path,'BBconvlstm_emotion.pkl'.format(t)))
    # save the entire model
    #torch.save(model, os.path.join(output_path,'convlstm.pkl'))
    
    #################################################################################
    ###########                 after training statistics                 ###########
    #################################################################################
    print ("*******************  Loss with time  **********************")
    fig00 = plt.figure()
    plt.plot(hist, 'r', label="Training loss")
    plt.xlabel('Iterations')
    plt.ylabel('MSE Error')
    plt.legend()
    fig00.savefig(os.path.join(output_path,'BBConvLSTM_train_mse_error.png'),dpi=150)
    
    print ("*******************  Loss with time (log)  **********************")
    fig01 = plt.figure()
    plt.plot(np.log(hist), 'r', label="Training loss")
    plt.xlabel('Iterations')
    plt.ylabel('Log mse error')
    plt.legend()
    plt.show()
    fig01.savefig(os.path.join(output_path,'BBConvLSTM_train_log_mse_error.png'),dpi=150)
    
    print ('*******************  evaluation matrix  *********************')
    # The prediction will be evaluated through RMSE against climatology
    
    # error score for temporal-spatial fields, without keeping spatial pattern
    def RMSE(x,y):
        """
        Calculate the RMSE. x is input series and y is reference series.
        It calculates RMSE over the domain, not over time. The spatial structure
        will not be kept.
        Parameter
        ----------------------
        x: input time series with the shape [time, lat, lon]
        """
        x_series = x.reshape(x.shape[0],-1)
        y_series = y.reshape(y.shape[0],-1)
        rmse = np.sqrt(np.mean((x_series - y_series)**2,1))
        rmse_std = np.sqrt(np.std((x_series - y_series)**2,1))
    
        return rmse, rmse_std
    
    # error score for temporal-spatial fields, keeping spatial pattern
    def MAE(x,y):
        """
        Calculate the MAE. x is input series and y is reference series.
        It calculate MAE over time and keeps the spatial structure.
        """
        mae = np.mean(np.abs(x-y),0)
        
        return mae
    
    def MSE(x, y):
        """
        Calculate the MSE. x is input series and y is reference series.
        """
        mse = np.mean((x-y)**2)
        
        return mse

    #################################################################################
    ########                           prediction                            ########
    #################################################################################
    print('##############################################################')
    print('###################  start prediction loop ###################')
    print('##############################################################')
    # forecast array
    pred_valance = np.zeros((batch_test_size, series_len, y_dim, x_dim),dtype=float)
    pred_arousal = np.zeros((batch_test_size, series_len, y_dim, x_dim),dtype=float)
    # calculate loss for each sample
    hist_valance = np.zeros(batch_test_size)
    hist_arousal = np.zeros(batch_test_size)
    for n in range(batch_test_size):
        # Clear stored gradient
        model.zero_grad()
        for timestep in range(sequence_len):
            x_input = np.stack((sample_1_test[n,timestep,:,:],
                                sample_2_test[n,timestep,:,:],
                                sample_3_test[n,timestep,:,:],
                                sample_4_test[n,timestep,:,:],
                                sample_5_test[n,timestep,:,:]))
            x_var_pred = torch.autograd.Variable(torch.Tensor(x_input).view(-1,input_channels,height,width),
                                                 requires_grad=False).to(device)
            # make prediction
            last_pred, _ = model(x_var_pred, timestep)
            # GPU data should be transferred to CPU
            pred_valance[n,timestep,:,:] = last_pred[0,0,:,:].cpu().data.numpy()
            pred_arousal[n,timestep,:,:] = last_pred[0,1,:,:].cpu().data.numpy()
        # compute the error for each sample
        hist_valance[n] = MSE(label_c_test_valance[n,:,:,:], pred_valance[n,:,:,:])
        hist_arousal[n] = MSE(label_c_test_arousal[n,:,:,:], pred_arousal[n,:,:,:]) 
    
    # save prediction as npz file
    np.savez_compressed(os.path.join(output_path,'BBConvLSTM_emotion_pred.npz'),
                        valance=pred_valance, arousal=pred_arousal)
    # plot the error
    print ("*******************  Loss with time  **********************")
    fig02 = plt.figure()
    plt.plot(hist_valance, 'r', label="Training loss - valance")
    plt.plot(hist_arousal, 'b', label="Training loss - arousal")
    plt.xlabel('Sample')
    plt.ylabel('MSE Error')
    plt.legend()
    plt.tight_layout()
    fig02.savefig(os.path.join(output_path,'BBConvLSTM_pred_mse_error.png'),dpi=150)
    
    #####################################################################################
    ########         visualization of prediction and implement metrics           ########
    #####################################################################################
    # compute mse
    mse_valance = MSE(label_c_test_valance, pred_valance)
    print(mse_valance)
    mse_arousal = MSE(label_c_test_arousal, pred_arousal)
    print(mse_arousal)
    # save output as csv file
    with open(os.path.join(output_path, "MSE_BBConvLSTM_emotion.csv"), "wt+") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["emotion prediction"])  # write header
        writer.writerow(["label valance"])
        writer.writerow([mse_valance])
        writer.writerow(["label arousal"])
        writer.writerow([mse_arousal])