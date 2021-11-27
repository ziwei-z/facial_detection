import os
import time
from datetime import datetime
from datetime import timedelta

import numpy as np
from numpy import mean
from numpy import std
from pandas.io.parsers import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import cv2

import utils_augment as aug

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from keras.datasets import mnist
#from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import Nadam

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.layer_utils import count_params


###################
## Process the data
###################

def load_data(fname, complete = True):
    """Loads data from fname.
    If complete = True, return only the subset of data with all the available facial features.
    """
    print(fname)
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    #print(df.count())  # prints the number of values for each column
    if complete:
        df = df.dropna()  # drop all rows that have missing values in them
    return(df)
    
def subset_data(preDF, test = False, n_train = 0, cols = None):
    '''Filter out the n_train rows and cols that we want to keep'''

    if cols:  # get a subset of columns
        preDF = preDF[list(cols) + ['Image']]

    X = np.vstack(preDF['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = preDF[preDF.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=0)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None
    
    if n_train > 0:
        # subset the first n_train rows
        X = X[:n_train,:]
        y = y[:n_train]

    return X, y

###### JM: SHOULD CONSOLIDATE OUR subset_data FUNCTIONS

def subset_data_JM(preDF, test = False, n_complete=500, n_partial=0, cols = None):
    
    '''Filter out the n_train rows and cols that we want to keep'''

    # of features per example, to distinguish complete from partial data
    df_feat_cnt = preDF.iloc[:, :-1].notnull().sum(axis=1)
    df_complete = preDF[df_feat_cnt == 30]
    df_partial = preDF[df_feat_cnt < 30]
    
       
    if cols:  # get a subset of columns
        df_complete = df_complete[list(cols) + ['Image']]
        df_partial = df_partial[list(cols) + ['Image']]


    X_c = np.vstack(df_complete['Image'].values) / 255.  # scale pixel values to [0, 1]
    X_p = np.vstack(df_partial['Image'].values) / 255.  # scale pixel values to [0, 1]
        
    X_c = X_c.astype(np.float32)
    X_p = X_p.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
      
        y_c = df_complete[df_complete.columns[:-1]].values
        y_p = df_partial[df_partial.columns[:-1]].values
              
        y_c = (y_c - 48) / 48  # scale target coordinates to [-1, 1]
        y_p = (y_p - 48) / 48  # scale target coordinates to [-1, 1]
        
        X_c, y_c = shuffle(X_c, y_c, random_state=0)  # shuffle train data
        X_p, y_p = shuffle(X_p, y_p, random_state=0)  # shuffle train data
        
        y_c, y_p = y_c.astype(np.float32), y_p.astype(np.float32)
        
    else:
        y_c, y_p = None, None
    
        
    if (n_complete > 0) or (n_partial>0):
        X_c, X_p = X_c[:n_complete,:], X_p[:n_partial,:]
        y_c, y_p = y_c[:n_complete], y_p[:n_partial]
        
    X = np.vstack((X_c, X_p))    
    y = np.vstack((y_c, y_p))      
                
    return X, y

#################

def split_data(params, dataX, dataY):
    '''select rows for train and test'''
    # shuffle the data
    np.random.seed(0)
    shuffle = np.random.permutation(np.arange(dataX.shape[0]))
    dataXshuffled, dataYshuffled = dataX[shuffle], dataY[shuffle]    

    #split the data into test and train
    train_size = np.array(dataX.shape[0] * (1 - params['validation_fraction'])).astype(int)
    trainX, trainY, testX, testY = dataXshuffled[:train_size], dataYshuffled[:train_size], dataXshuffled[train_size:], dataYshuffled[train_size:]
    
    return(trainX, trainY, testX, testY)

def split_data_by_completeness(df, cols):
    '''Split the data into those rows of df with all the features and those 
    missing some, but contain all cols'''

    # missing data per row vec:
    
    missing_data_vec = df.isna().sum(axis = 1) > 0
    
    # full data:
    df_full = df[missing_data_vec == 0]
    
    # incomplete data:
    df_incomplete = df[missing_data_vec == 1]
    
    # of the imcomplete data, filter out those with complete cols
    
    df_incomplete = df_incomplete[list(cols) + ['Image']]
    df_incomplete = df_incomplete.dropna()
    
    return(df_full, df_incomplete)    

def split_xy(preDF, test = False,):
    '''Split the data into X and Y components, 
    transform their scale from 0 to 95 to -1 to 1,
    suffle the returned data'''
    
    X = np.vstack(preDF['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = preDF[preDF.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=0)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return(X, y)



def make_square(tmp_data):
    '''Per image, make the data square'''   
    
    tmp_data_square = tmp_data.reshape(-1,96,96,1)

    return tmp_data_square



def make_flat(tmp_data):
    '''Per image, flatten the data'''
    
    tmp_data_flat = tmp_data.reshape((tmp_data.shape[0], -1)) 

    return tmp_data_flat



def load2d(test=False,cols=None):
   
    re = load(test, cols)
    
    X = re[0].reshape(-1,96,96,1)
    y = re[1]

    return X, y

def build_param_str(params, n = 0):
    tmp_str = []
    for key, value in params.items():
        if (key != 'augment'):
            tmp_str.append(str(value).replace('.', ''))
        else:
            aug_string = []
            for key, value in params['augment'].items():
                aug_string.append(key[0].upper() + str(1*value))
            tmp_str.append("".join(aug_string))
    param_str = "_"
    param_str += "_".join(tmp_str)
    param_str += "_" + str(n)
    return(param_str)

###############
## Plot Results
###############

def plot_sample(X,y,axs):
    '''
    kaggle picture is 96 by 96
    y is rescaled to range between -1 and 1
    '''
    
    axs.imshow(X.reshape(96,96),cmap="gray")
    axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48, s = 3, color='red')
    

def plot_compare(X, y, y_pred, y_means, axs):
    '''
    kaggle picture is 96 by 96
    y is rescaled to range between -1 and 1
    '''
    
    axs.imshow(X.reshape(96,96),cmap="gray")
    #axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48, s = 10, marker = 's', color='green')
    axs.scatter(48*y_pred[0::2]+ 48,48*y_pred[1::2]+ 48, s = 30, color='red')
    axs.scatter(48*y_means[0::2]+ 48,48*y_means[1::2]+ 48, s = 30, color='blue')
    axs.scatter(48*y[0::2]+ 48,48*y[1::2]+ 48, s = 40, marker = 's', color='yellow')
    axs.axis('off')
    return(axs)
    
def plot_loss(hist,name,plt,RMSE_TF=False):
    '''
    RMSE_TF: if True, then RMSE is plotted with original scale 
    '''
    loss = hist['loss']
    val_loss = hist['val_loss']
    if RMSE_TF:
        loss = np.sqrt(np.array(loss))*48 
        val_loss = np.sqrt(np.array(val_loss))*48 
        
    plt.plot(loss,"--",linewidth=3,label="train:"+name)
    plt.plot(val_loss,linewidth=3,label="val:"+name)
    
def plot_boxplot_with_means(tmp_data, title_str = None, x_str = None, min_x = 0, max_x = 5):
    '''input is a dataFrame. Calculate the mean per column and then plot the boxplot and overlay the means'''
    
    # calculate the column means
    mean_df = pd.DataFrame(
        tmp_data.\
        mean(axis = 0).\
        rename_axis('feature'),
        columns = ['mean']
        ).reset_index()

    # plot the data:
    plt.figure(num = 1, figsize=(10, 10))  
    plt.xlim(min_x, max_x)
    plt.title(title_str)
    plt.xlabel(x_str)
    #sns.set(style="darkgrid")
    sns.boxplot(data = tmp_data, orient = 'h', palette = ['b', 'r'])
    sns.scatterplot(y = mean_df['feature'], x = mean_df['mean'], s = 100, color = ['k'])
    plt.axvline(x = 0, color = 'k')
    plt.show()
    
def summarize_diagnostics_kfolds(histories, modelType = '', title_string = ''):
    '''plot diagnostic learning curves''' 
    for i in range(len(histories)):
        # plot loss
        plt.subplot(1, 1, 1)
        plt.title('Loss: '+ modelType + title_string)
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.yscale('log')
        #plt.ylim(0.0005, 0.01)
        plt.legend(["train", "validate"])
    plt.show()

# def summarize_diagnostics(history_df, modelType = '', title_string = ''):
#     '''plot diagnostic learning curves''' 

#     max_epochs = history_df['epoch'].iloc[-1] + 1
#     val_loss   = np.min(history_df['val_loss'])
#     train_loss = np.min(history_df['loss'])
#     loss_ratio = val_loss / train_loss
    
#     # find the lower bound for the chart:
#     lower_limit = 10**np.floor(
#         np.log10(
#             np.minimum(val_loss,train_loss)
#             )
#         )
    
#     tmp_str = f'Epochs: {max_epochs} MSE Val: {val_loss:0.5f} MSE Train: {train_loss:0.5f} Loss Ratio: {loss_ratio:0.5f}'
    
#     plt.subplot(1, 1, 1)
#     plt.suptitle(title_string + "\n")    
#     plt.title(tmp_str)
#     #plt.title(tmp_str, y = -.5) # title below image
#     #plt.title(tmp_str)
#     #print(tmp_str)
    
    
#     plt.plot(history_df['epoch']+1, history_df['loss'], color='blue', label='train')
#     plt.plot(history_df['epoch']+1, history_df['val_loss'], color='orange', label='test')
#     plt.xlabel("epoch")
#     plt.ylabel("MSE (image scaled to -1:1)")
#     plt.yscale('log')
#     if lower_limit < 0.01:
#         plt.ylim(lower_limit, 0.01)
#     plt.legend(["train", "validate"])
    
#     #plt.text(15, -0.01, tmp_str)
#     plt.text(tmp_str, y = -.01)
#     plt.show()
#     #plt.text(tmp_str, y = -.01)
#     print("foobar")
    


def trainable_count(model = None):
    if model is None:
        pass
    else:
        return count_params(model.trainable_weights)

    
def return_diagnostics(model, history_df, trainX, trainY, valX, valY):
    max_epochs = history_df['epoch'].iloc[-1] + 1
    #val_loss   = np.min(history_df['val_loss'])
    val_loss = model.evaluate(valX, valY)
    
    val_rmse = val_loss**0.5
    val_rmse_px = val_rmse*48
    #train_loss = np.min(history_df['loss'])
    train_loss = model.evaluate(trainX, trainY)
    train_rmse = train_loss**0.5
    train_rmse_px = train_rmse*48
    loss_ratio = val_rmse / train_rmse  
    model_name = model.name
    t_params = trainable_count(model)
    print(train_rmse_px, val_rmse_px)
    
    diag_dict = {"max_epochs": max_epochs,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_rmse_px": val_rmse_px,
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "train_rmse_px": train_rmse_px,
                "loss_ratio": loss_ratio, 
                "model_name": model_name,
                "t_params": t_params
                }
    
    return diag_dict



# def print_diagnostics(m_list, d_list, t_list):

#     col_lab = "\t\t\t     "
#     hyp_str = "_"*29
#     only_col = " "*29
#     for m in m_list:
            
#             col_lab += "|   " + m.name + "   "
#             hyp_str += "_"*14
#             only_col += "|" + 13*" "
#     col_lab += "|"
#     hyp_str += "_"
#     only_col += "|"
    
#     print("\n\nModel Comparison\n")
    
#     print(col_lab)
#     print(hyp_str)

#     v_rmse_list = []
#     v_rmse_str = "    Validation RMSE (pixels) "
    
#     t_rmse_list = []
#     t_rmse_str = "         Train RMSE (pixels) " 
    
    
    
    
    
#     test_rmse_list = []
#     test_rmse_str = "          Test RMSE (pixels) " 
    
    
#     lr_vt_list = []
#     lr_vt_str = "      Loss Ratio (Val/Train) "
    
#     lr_tv_list = []
#     lr_tv_str = "       Loss Ratio (Test/Val) "
    
#     e_list = []
#     e_str = "\t\t      Epochs "
    
#     tp_list = []
#     tp_str = "        Trainable Parameters "
    

#     for i in range(len(m_list)):
#         v_rmse_list.append(d_list[i]['val_rmse_px'])
#         v_rmse_str += "|{" + str(i) + ": >12.2f} "

#         t_rmse_list.append(d_list[i]['train_rmse_px'])
#         t_rmse_str += "|{" + str(i) + ": >12.2f} " 

#         test_rmse_list.append(t_list[i])
#         test_rmse_str += "|{" + str(i) + ": >12.2f} " 
                
#         lr_vt_list.append(d_list[i]['loss_ratio'])
#         lr_vt_str += "|{" + str(i) + ": >12.2f} " 
        
#         lr_tv_list.append(t_list[i]/d_list[i]['val_rmse_px'])
#         lr_tv_str += "|{" + str(i) + ": >12.2f} "         
        

#         e_list.append(d_list[i]['max_epochs'])
#         e_str += "|{" + str(i) + ": >12,.0f} " 

#         tp_list.append(d_list[i]['t_params'])
#         tp_str += "|{" + str(i) + ": >12,.0f} "     

#     v_rmse_str += "|"
#     t_rmse_str += "|"  
#     test_rmse_str += "|"
#     lr_vt_str += "|"  
#     e_str += "|"
#     tp_str += "|"
  


#     print(t_rmse_str.format(*t_rmse_list))
#     print(v_rmse_str.format(*v_rmse_list)) 
#     print(test_rmse_str.format(*test_rmse_list))  
    
#     print(hyp_str)
#     print(lr_vt_str.format(*lr_vt_list))
#     print(lr_tv_str.format(*lr_tv_list))
    
#     print(only_col)
#     #print("\n")
    
     
#     print(e_str.format(*e_list))
#     print(tp_str.format(*tp_list))
#     print(hyp_str)

    
def print_diagnostics(m_list, d_list, t_list):

    col_lab = "\t\t\t     "
    hyp_str = "_"*29
    only_col = " "*29
    for m in m_list:
        if len(m.name) == 8:
            col_lab += "|   " + m.name + "  "
        elif len(m.name) == 9:
            col_lab += "|  " + m.name + "  "
        else:    
            col_lab += "|   " + m.name + "   "
        hyp_str += "_"*14
        only_col += "|" + 13*" "
    col_lab += "|"
    hyp_str += "_"
    only_col += "|"
    
    print("\n\nModel Comparison\n")
    
    print(col_lab)
    print(hyp_str)

    v_rmse_list = []
    v_rmse_str = "    Validation RMSE (pixels) "
    
    t_rmse_list = []
    t_rmse_str = "         Train RMSE (pixels) " 
    
    
    
    
    
    test_rmse_list = []
    test_rmse_str = "          Test RMSE (pixels) " 
    
    
    lr_vt_list = []
    lr_vt_str = "      Loss Ratio (Val/Train) "
    
    lr_tv_list = []
    lr_tv_str = "      Loss Ratio (Test/Train)"
    
    e_list = []
    e_str = "\t\t      Epochs "
    
    tp_list = []
    tp_str = "        Trainable Parameters "
    

    for i in range(len(m_list)):
        v_rmse_list.append(d_list[i]['val_rmse_px'])
        v_rmse_str += "|{" + str(i) + ": >12.2f} "

        t_rmse_list.append(d_list[i]['train_rmse_px'])
        t_rmse_str += "|{" + str(i) + ": >12.2f} " 

        test_rmse_list.append(t_list[i])
        test_rmse_str += "|{" + str(i) + ": >12.2f} " 
                
        #lr_vt_list.append(d_list[i]['loss_ratio'])
        
        lr_vt_list.append(d_list[i]['val_rmse_px']/d_list[i]['train_rmse_px'])
        lr_vt_str += "|{" + str(i) + ": >12.2f} " 
        
        #lr_tv_list.append(t_list[i]/d_list[i]['val_rmse_px']         )
        lr_tv_list.append(t_list[i]/d_list[i]['train_rmse_px'])
        
        lr_tv_str += "|{" + str(i) + ": >12.2f} "         
        

        e_list.append(d_list[i]['max_epochs'])
        e_str += "|{" + str(i) + ": >12,.0f} " 

        tp_list.append(d_list[i]['t_params'])
        tp_str += "|{" + str(i) + ": >12,.0f} "     

    v_rmse_str += "|"
    t_rmse_str += "|"  
    test_rmse_str += "|"
    lr_vt_str += "|"  
    e_str += "|"
    tp_str += "|"
  


    print(t_rmse_str.format(*t_rmse_list))
    print(v_rmse_str.format(*v_rmse_list)) 
    print(test_rmse_str.format(*test_rmse_list))  
    
    print(hyp_str)
    print(lr_vt_str.format(*lr_vt_list))
    print(lr_tv_str.format(*lr_tv_list))
    
    print(only_col)
    #print("\n")
    
     
    print(e_str.format(*e_list))
    print(tp_str.format(*tp_list))
    print(hyp_str)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def summarize_diagnostics(history_df, modelType = '', title_string = ''):
    '''plot diagnostic learning curves''' 

    max_epochs = history_df['epoch'].iloc[-1] + 1
    val_loss   = np.min(history_df['val_loss'])
    val_rmse = val_loss**0.5
    val_rmse_px = val_rmse*48
    train_loss = np.min(history_df['loss'])
    train_rmse = train_loss**0.5
    train_rmse_px = train_rmse*48
    loss_ratio = val_loss / train_loss
    
    # find the lower bound for the chart:
    lower_limit = 10**np.floor(
        np.log10(
            np.minimum(val_loss,train_loss)
            )
        )
  
    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 4))
    
    fig.suptitle(title_string)    
    
    ax.plot(history_df['epoch']+1, history_df['loss'], color='blue', label='train')
    ax.plot(history_df['epoch']+1, history_df['val_loss'], color='orange', label='test')
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (image scaled to -1:1)")
    ax.set_yscale('log')
    if lower_limit < 0.01:
        ax.set_ylim(lower_limit, 0.01)
    ax.legend(["train", "validate"])
    
   
    plt.show()
    
#     print(f"\nEpochs: {max_epochs}")
#     print(f"Loss Val: {val_loss:0.4f}, Loss Train {train_loss:0.4f}")
#     print(f"Loss Ratio: {loss_ratio:0.3f}\n") 
#     print(f"RMSE Val: {val_rmse:0.4f}, RMSE Train {train_rmse:0.4f}")
#     print(f"Pixel RMSE Val: {val_rmse_px:0.2f}, Pixel RMSE Train {train_rmse_px:0.2f}\n")
   
   
    
def summarize_performance(scores, modelType = '', title_string = ''):
    '''This is relevant for multiple kfolds'''
    print(f'MSE: mean={mean(scores):.5f} std={std(scores):.5f}, n={len(scores)}')
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.title('Loss: '+ modelType + title_string)
    plt.show()

def plot_rmse(rmse_df):
    plt.figure(num = 1, figsize=(10, 10))  
    plt.xlim(0, 5)
    plt.xlabel('RMSE (pixels)')
    #sns.set(style="darkgrid")
    sns.boxplot(data = rmse_df, orient = 'h', palette = ['b', 'r'])
    plt.title('Distribution of RMSE per image and facial feature')
    plt.show()
    return()

    
def plot_multiple_images(X_square, y, y_pred, y_means, image_vec):
    '''Plot a sequence of images, with the facial features'''
    
    print('Yellow = Truth, Red: Model, Blue: Average over all images')
    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(hspace=0.13,wspace=0.0001,
                        left=0,right=1,bottom=0, top=1)

    Npicture = len(image_vec)
    n_cols = 3
    n_rows = int(Npicture/n_cols)
    count = 1
    for irow in range(Npicture):
        ipic = image_vec[irow]
        ax = fig.add_subplot(int(Npicture/3) , 3, count,xticks=[],yticks=[])        
        ax = plot_compare(X_square[ipic], y[ipic], y_pred[ipic], y_means, ax)
        ax.set_title("Image "+ str(ipic))
        count += 1

    plt.show()
    
###########################
## Calculations on the data
###########################

def calc_rmse_pixels(y, y_pred, facial_feature_vec):
    rmse_df = pd.DataFrame(
        48 * np.sqrt((y - y_pred)**2),
        columns = facial_feature_vec)
    return(rmse_df)

def calc_model_rmse_pixels(y, y_pred, facial_feature_vec, sub_facial_feature_vec = None):
    '''Calculate the RMS for the entire model, allowing to subset on some features'''
    # per element, calc the MSE:
    mse_df = pd.DataFrame(
        (y - np.array(y_pred))**2,
        columns = facial_feature_vec) 
    
    # filter out the columns we care about:
    if sub_facial_feature_vec != None:
        mse_df = mse_df[sub_facial_feature_vec]
    
    # turn the dataFrame into a vector calc the mean and the sqrt:
    model_rms = 48 * np.sqrt(
        np.mean(
            mse_df.values.reshape(-1)
        ))     # turn the data frame into a vector
    
    return(model_rms)


def evaluate_model(trainX, trainY, testX, testY, params, model = None, history_name = None):
    '''Evaluate the model'''
    scores, histories = list(), list()
    # run a single pass
    print (f'Starting run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')
    cv_tic = time.perf_counter()
    # define model
    test_model = model

    # save the history (if the file is already there, delete it, otherwise keep going)
    #if os.path.exists(history_name):
    #    os.remove(history_name)
    csv_logger = CSVLogger(history_name, append=False, separator=',')

    # JM: NOTE ADDITION TO CALLBACKS
    early_stopping_cb = EarlyStopping(patience=50,restore_best_weights=True)
      
    # fit model
    history = test_model.fit(trainX, trainY, epochs=params['epochs'], validation_data=(testX, testY), verbose=0, callbacks=[csv_logger, early_stopping_cb])

    # evaluate test_model
    mse = test_model.evaluate(testX, testY, verbose=0)
    cv_toc = time.perf_counter()
    print(f'mse: {mse:.4f}, Run time = {str(timedelta(seconds = cv_toc - cv_tic))}')
    
    # stores scores
    scores.append(mse)
    histories.append(history)

    return (scores, histories)

def evaluate_dual_models(trainX, train_metaX, trainY,  testX, test_metaX, testY, params, model = None, history_name = None):
    '''Evaluate the model'''
    scores, histories = list(), list()
    # run a single pass
    print (f'Starting run at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')
    cv_tic = time.perf_counter()
    # define model
    test_model = model

    # save the history (if the file is already there, delete it, otherwise keep going)
    #if os.path.exists(history_name):
    #    os.remove(history_name)
    csv_logger = CSVLogger(history_name, append=False, separator=',')

    # JM: NOTE ADDITION TO CALLBACKS
    early_stopping_cb = EarlyStopping(patience=50,restore_best_weights=True)
      
    # fit model
    history = test_model.fit([trainX, train_metaX], trainY, epochs=params['epochs'], validation_data=([testX, test_metaX], testY), verbose=0, callbacks=[csv_logger, early_stopping_cb])

    # evaluate test_model
    mse = test_model.evaluate([testX, test_metaX], testY, verbose=0)
    cv_toc = time.perf_counter()
    print(f'mse: {mse:.4f}, Run time = {str(timedelta(seconds = cv_toc - cv_tic))}')
    
    # stores scores
    scores.append(mse)
    histories.append(history)

    return (scores, histories)


    
####################
## Set up the models
####################

# def run_model(model, trainX, trainY, testX, testY, params, build_model, model_dir_name, param_str):
#     '''Function to fit or reload a model'''
    
#     file_name = model_dir_name + model.name + param_str
#     history_name = file_name + "_history.csv"
#     print(file_name)

#     if build_model:
#         tic = time.perf_counter()
#         evaluate_model(trainX, trainY, testX, testY, params, model = model, history_name = history_name)

#         print('Saving:', file_name)
#         model.save(file_name)

#         toc = time.perf_counter()
#         print(f'Run time = {str(timedelta(seconds = toc - tic))}')

#     else:
#         model = load_model(file_name)

#     # load the history
#     if os.path.exists(history_name):
#         title_string = model.name + param_str
#         history_df = pd.read_csv(history_name)
#         summarize_diagnostics(history_df, model.name, title_string)
#     else:
#         print(f'There is no history file named {history_name}')

#     return(model)




def run_model(model, trainX, trainY, testX, testY, params, build_model, model_dir_name, param_str):
    '''Function to fit or reload a model'''
    
    file_name = model_dir_name + model.name + param_str
    history_name = file_name + "_history.csv"
    print(file_name)

    if build_model:
        tic = time.perf_counter()
        evaluate_model(trainX, trainY, testX, testY, params, model = model, history_name = history_name)

        print('Saving:', file_name)
        model.save(file_name)
        model = load_model(file_name) ################## not sure should be added

        toc = time.perf_counter()
        print(f'Run time = {str(timedelta(seconds = toc - tic))}')
             
    else:
        model = load_model(file_name)
        
    # load the history
    if os.path.exists(history_name):
        #title_string = model.name + param_str 
        title_string = model.name 
        history_df = pd.read_csv(history_name)
        summarize_diagnostics(history_df, model.name, title_string)
    else:
        print(f'There is no history file named {history_name}')

    return(model  , history_df)



def run_dual_models(model, trainX, train_metaX, trainY,  testX, test_metaX, testY, params, build_model, model_dir_name, param_str):
    '''Function to fit or reload a model'''
    
    file_name = model_dir_name + model.name + param_str
    history_name = file_name + "_history.csv"
    print(file_name)

    
    if build_model:
        tic = time.perf_counter()
        evaluate_dual_models(trainX, train_metaX, trainY,  testX, test_metaX, testY, params, model = model, history_name = history_name)
                              
        print('Saving:', file_name)
        model.save(file_name)
        model = load_model(file_name) ################## not sure should be added

        toc = time.perf_counter()
        print(f'Run time = {str(timedelta(seconds = toc - tic))}')
             
    else:
        model = load_model(file_name)
        
    # load the history
    if os.path.exists(history_name):
        title_string = model.name + param_str 
        history_df = pd.read_csv(history_name)
        summarize_diagnostics(history_df, model.name, title_string)
    else:
        print(f'There is no history file named {history_name}')

    return(model  , history_df)









def model1(num_y_features):
    model = Sequential(name = 'model1')
    model.add(Flatten())
    model.add(Dense(9216, activation = 'relu'))
    model.add(Dense(num_y_features))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss = "mean_squared_error", optimizer = opt)
    return(model)

def model2(num_y_features):
    model = Sequential(name = 'model2')
    model.add(Flatten())
    model.add(Dense(9216, activation = 'relu'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(num_y_features))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss = "mean_squared_error", optimizer = opt)
    return(model)

def model3(num_y_features):
    model = Sequential(name = 'model3')
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation = 'relu'))
    model.add(Dense(num_y_features))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss = "mean_squared_error", optimizer = opt)
    print(model.summary())
    return(model)

def model4(num_y_features):
    model = Sequential(name = 'model4')
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(num_y_features))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss = "mean_squared_error", optimizer = opt)
    print(model.summary())
    return(model)

def model5(num_y_features):
    model = Sequential(name = 'model5')
    model.add(Conv2D(32, (3, 3), activation='relu', 
                     kernel_initializer='he_uniform', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(num_y_features))
    #opt = SGD(lr=0.01, momentum=0.9, nesterov = True)
    #opt = RMSprop(lr=0.001, rho=0.9)
    opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999)
    #opt = Adamax()
    #opt = Nadam()
    model.compile(loss = "mean_squared_error", optimizer = opt)
    #model.name = 'model5'
    print(model.summary())
    return(model)

def model6(num_y_features):
    model = Sequential(name = 'model6')
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(96, 96, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(num_y_features))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss = "mean_squared_error", optimizer = opt)
    print(model.summary())
    return(model)

def model_jm3(num_y_features):
    '''Joes model 3'''
    d_rate = 0.2
    n_nodes = 125
    
    model = Sequential(name = 'model_jm3')
    model.add(Flatten())
    
    model.add(BatchNormalization())
    
    
    #model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    
    
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(BatchNormalization())
    
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(BatchNormalization())
    
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(BatchNormalization())
    
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(BatchNormalization())
    
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(BatchNormalization())
    
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(Dropout(rate = d_rate))
    model.add(BatchNormalization())
   
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(Dropout(rate = d_rate))
    model.add(BatchNormalization())
    
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(Dropout(rate = d_rate))
    model.add(BatchNormalization())
  
    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(Dropout(rate = d_rate))
    model.add(BatchNormalization())

    model.add(Dense(n_nodes, activation = 'elu', kernel_initializer = "he_normal"))
    model.add(Dropout(rate = d_rate))
    model.add(BatchNormalization())
    
    model.add(Dense(num_y_features))
    #opt = SGD(lr=0.01, momentum=0.9)
    
    opt = Nadam(learning_rate=.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    
    model.compile(loss = "mean_squared_error", optimizer = opt)
    return(model)

def model_jm4(num_y_features):
    '''Joes model 4'''

    d_rate = 0.5
    n_nodes = 125
    

    model = Sequential(name = 'model_jm4')
    
    # trying to condense the data somewhat, but using a lot of filters to start and very large convolution kernel 
    #model.add(Conv2D(128, 12, activation='relu', kernel_initializer='he_uniform', padding="same", input_shape=(96, 96, 1)))
    model.add(Conv2D(128, 12, activation='selu', kernel_initializer='lecun_normal', padding="same", input_shape=(96, 96, 1)))
    model.add(BatchNormalization())  
    model.add(MaxPooling2D((2, 2)))
    
    
    model.add(Conv2D(128, 2, activation='selu', kernel_initializer='lecun_normal', padding="same"))
    model.add(BatchNormalization())  
    model.add(Conv2D(128, 2, activation='selu', kernel_initializer='lecun_normal', padding="same"))
    model.add(BatchNormalization())     
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())             

    model.add(Conv2D(256, 2, activation='selu', kernel_initializer='lecun_normal', padding="same"))
    model.add(BatchNormalization())  
    model.add(Conv2D(256, 2, activation='selu', kernel_initializer='lecun_normal', padding="same"))
    model.add(BatchNormalization())     
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())             
              
    model.add(Flatten())   
    model.add(BatchNormalization())            
    
    #model.add(Dense(256, activation = 'relu')) 
    model.add(Dense(256, activation = 'selu', kernel_initializer='lecun_normal'))   
    model.add(BatchNormalization())    
    model.add(Dropout(rate = d_rate))
            
#     model.add(Dense(128, activation = 'selu', kernel_initializer='lecun_normal'))     
#     model.add(BatchNormalization())    
#     model.add(Dropout(rate = d_rate))
   
    model.add(Dense(64, activation = 'selu', kernel_initializer='lecun_normal'))   
    model.add(BatchNormalization())    
    model.add(Dropout(rate = d_rate))
              
    model.add(Dense(num_y_features))
                        
    opt = Nadam(learning_rate=.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")          
    #opt = SGD(lr=0.01, momentum=0.9)
    
    model.compile(loss = "mean_squared_error", optimizer = opt)
    print(model.summary())
    return(model)

def model_d3(num_y_features, n_image, n_meta):
    
    # dual path of Joe's model 3
    
    d_rate = 0.2
    n_nodes = 125
        
    input_meta  = Input(shape = [n_meta], name = 'meta')
    input_image = Input(shape = [n_image], name = 'image')

#     hidden1a = Dense(n_nodes, activation = 'relu')(input_meta)
#     hidden1a_b = BatchNormalization()(hidden1a)
#     hidden2a = Dense(n_nodes,  activation = 'relu')(hidden1a_b)
#     hidden2a_b = BatchNormalization()(hidden2a)
#     hidden3a = Dense(n_nodes, activation = 'relu')(hidden2a_b)
#     hidden3a_b = BatchNormalization()(hidden3a)
#     hidden4a = Dense(n_nodes,  activation = 'relu')(hidden3a_b)
#     hidden4a_b = BatchNormalization()(hidden4a)
#     hidden5a = Dense(n_nodes, activation = 'relu')(hidden4a_b)
#     hidden5a_b = BatchNormalization()(hidden5a)

#     hidden6a = Dense(n_nodes, activation = 'relu')(hidden5a_b)
#     hidden6a_d = Dropout(rate = d_rate)(hidden6a)
#     hidden6a_b = BatchNormalization()(hidden6a_d)
#     hidden7a = Dense(n_nodes, activation = 'relu')(hidden6a_b)
#     hidden7a_d = Dropout(rate = d_rate)(hidden7a)
#     hidden7a_b = BatchNormalization()(hidden7a_d)
#     hidden8a = Dense(n_nodes, activation = 'relu')(hidden7a_b)
#     hidden8a_d = Dropout(rate = d_rate)(hidden8a)
#     hidden8a_b = BatchNormalization()(hidden8a_d)
#     hidden9a = Dense(n_nodes, activation = 'relu')(hidden8a_b)
#     hidden9a_d = Dropout(rate = d_rate)(hidden9a)
#     hidden9a_b = BatchNormalization()(hidden9a_d)

    hidden1b = Dense(n_nodes, activation = 'relu')(input_image)
    hidden1b_b = BatchNormalization()(hidden1b)
    hidden2b = Dense(n_nodes,  activation = 'relu')(hidden1b_b)
    hidden2b_b = BatchNormalization()(hidden2b)
    hidden3b = Dense(n_nodes, activation = 'relu')(hidden2b_b)
    hidden3b_b = BatchNormalization()(hidden3b)
    hidden4b = Dense(n_nodes,  activation = 'relu')(hidden3b_b)
    hidden4b_b = BatchNormalization()(hidden4b)
    hidden5b = Dense(n_nodes, activation = 'relu')(hidden4b_b)
    hidden5b_b = BatchNormalization()(hidden5b)

    hidden6b = Dense(n_nodes, activation = 'relu')(hidden5b_b)
    hidden6b_d = Dropout(rate = d_rate)(hidden6b)
    hidden6b_b = BatchNormalization()(hidden6b_d)
    hidden7b = Dense(n_nodes, activation = 'relu')(hidden6b_b)
    hidden7b_d = Dropout(rate = d_rate)(hidden7b)
    hidden7b_b = BatchNormalization()(hidden7b_d)
    hidden8b = Dense(n_nodes, activation = 'relu')(hidden7b_b)
    hidden8b_d = Dropout(rate = d_rate)(hidden8b)
    hidden8b_b = BatchNormalization()(hidden8b_d)
    hidden9b = Dense(n_nodes, activation = 'relu')(hidden8b_b)
    hidden9b_d = Dropout(rate = d_rate)(hidden9b)
    hidden9b_b = BatchNormalization()(hidden9b_d)


   # concat  = concatenate([hidden9a_b, hidden9b_b])
    concat  = concatenate([input_meta, hidden9b_b])

    hidden1c = Dense(n_nodes,  activation = 'relu')(concat)
    hidden1c_d = Dropout(rate = d_rate)(hidden1c)
    hidden1c_b = BatchNormalization()(hidden1c_d)
    hidden2c = Dense(n_nodes,  activation = 'relu')(hidden1c_b)
    hidden2c_d = Dropout(rate = d_rate)(hidden2c)
    hidden2c_b = BatchNormalization()(hidden2c_d)

    output  = Dense(num_y_features)(hidden2c_b)

    model = Model(inputs = [input_meta, input_image], outputs = [output], name = 'd3')
    #opt = SGD(lr=0.01, momentum=0.9)
    opt = Nadam(learning_rate=.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    model.compile(loss = "mean_squared_error", optimizer = opt)

    return(model)

def model_d4(num_y_features, n_image, n_meta):
    
    # dual path of Joe's model 4

    d_rate = 0.2
    n_nodes = 125
        
    input_image = Input(shape = [n_image], name = 'image')
    input_meta  = Input(shape = [n_meta], name = 'meta')


    return(model)

###########################
# Set up the error analysis
###########################


def calc_rmse(y_test, y_pred, facial_feature_vec):
    # cacluate the pairwise rms:
    rmse_df = pd.DataFrame(
        48 * np.sqrt((y_test - y_pred)**2),
        columns = facial_feature_vec)
    return(rmse_df)

def calc_angle(x):
    delta_y = x.left_eye_center_y - x.right_eye_center_y
    delta_x = x.left_eye_center_x - x.right_eye_center_x
    slope = delta_y / delta_x
    #return(np.arctan(slope) * 180 / np.pi)
    return(np.degrees(np.arctan2(delta_y, delta_x)))


def calc_eye_distance(x):
    delta_y = x.left_eye_center_y - x.right_eye_center_y
    delta_x = x.left_eye_center_x - x.right_eye_center_x
    distance = 48 * np.sqrt(delta_y**2 + delta_x**2)
    return(distance)

def calc_mid_point_between_eyes(x):
    mid_x = 48 * ((x.left_eye_center_x + x.right_eye_center_x)/2)
    mid_y = 48 * (1 + (x.left_eye_center_y + x.right_eye_center_y)/2)
    return(mid_x, mid_y)

def calc_distances(x):
    # distance between eyes:
    delta_y = x.left_eye_center_y - x.right_eye_center_y
    delta_x = x.left_eye_center_x - x.right_eye_center_x
    d_Leye_Reye = 48 * np.sqrt(delta_y**2 + delta_x**2)
    
    # distance between left eye and nose tip:
    delta_y = x.left_eye_center_y - x.nose_tip_y
    delta_x = x.left_eye_center_x - x.nose_tip_x
    d_Leye_nose = 48 * np.sqrt(delta_y**2 + delta_x**2)

    # distance between right eye and nose tip:
    delta_y = x.right_eye_center_y - x.nose_tip_y
    delta_x = x.right_eye_center_x - x.nose_tip_x
    d_Reye_nose = 48 * np.sqrt(delta_y**2 + delta_x**2)

    # distance between left eye and mouth center bottom lip:
    delta_y = x.left_eye_center_y - x.mouth_center_bottom_lip_y
    delta_x = x.left_eye_center_x - x.mouth_center_bottom_lip_x
    d_Leye_mouth = 48 * np.sqrt(delta_y**2 + delta_x**2)

    # distance between right eye and mouth center bottom lip:
    delta_y = x.right_eye_center_y - x.mouth_center_bottom_lip_y
    delta_x = x.right_eye_center_x - x.mouth_center_bottom_lip_x
    d_Reye_mouth = 48 * np.sqrt(delta_y**2 + delta_x**2)

    return(d_Leye_Reye, d_Leye_nose, d_Reye_nose, d_Leye_mouth, d_Reye_mouth)    

def plot_error_analysis(rmse_df, meta_rms_df):
    
    meta_rms_df['RMS per Image'] = meta_rms_df['rms_state'] # we apply this here because it's earsier to change the kdeplot legend title.
    # create the summary plots:
    
    fig, axes = plt.subplots(4, 2, figsize = (15,30), gridspec_kw={'hspace': 0.25, 'wspace': 0.45})
    axes = axes.flatten()
    
    # plot the RMSE distributions per facial feature
    axes_indx = 0
    axes[axes_indx].set_xlim(0, 6)
    axes[axes_indx].set_xlabel('RMSE (pixels)')
    sns.boxplot(data = rmse_df, orient = 'h', palette = ['b', 'r'], ax = axes[axes_indx])
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Distribution of RMSE per image and facial feature')
    #plt.show()
    
    # compare the angle between the eyes (test vs. pred)
    axes_indx = 2
    sns.scatterplot(x = meta_rms_df['angle_test'], y =  meta_rms_df['angle_pred'], hue = meta_rms_df['RMS per Image'], alpha = 0.3, ax = axes[axes_indx])
    axes[axes_indx].set_xlabel('Angle between left eye and right eye (test, degrees)')
    axes[axes_indx].set_ylabel('Angle between left eye and right eye (predicted, degrees)')
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Tilt Error: Predicted angle between eyes vs. test angle')
    axes[axes_indx].axhline(0, ls='-', linewidth = 1, color = 'k')
    axes[axes_indx].axvline(0, ls='-', linewidth = 1, color = 'k')
    axes[axes_indx].set_ylim(-20, 20)
    axes[axes_indx].set_xlim(-20, 20)

    # plot the distance between the eyes.  This will be a proxy for warping
    axes_indx = 3
    sns.kdeplot(data = meta_rms_df, x = 'd_Leye_Reye', hue = "RMS per Image",  ax = axes[axes_indx])    
    axes[axes_indx].set_xlabel('Distance between eye centers (pixels)')
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Warping Error: Distance between the eyes')

    # plot the distance between each eye and the nose.  This will be a proxy for head turn
    axes_indx = 4
    sns.kdeplot(data = meta_rms_df, x = 'head_turn', hue = "RMS per Image",  ax = axes[axes_indx])    
    axes[axes_indx].set_xlabel('Difference of the distance between the eyes and nose (pixels)')
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Head Turn Error: Distance between each eye and the nose')

    # plot the location of the mid point between the eyes.  This will be a proxy for translation vertically or horizontally
    axes_indx = 5
    sns.scatterplot(x = meta_rms_df['mid_x'], y =  meta_rms_df['mid_y'], hue = meta_rms_df['RMS per Image'], alpha = 0.3, ax = axes[axes_indx])
    axes[axes_indx].set_xlabel('Pixels from center of the image')
    axes[axes_indx].set_ylabel('Pixels from the top of the image')
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Translation Error: Location of the midpoint between the eyes')
    #axes[axes_indx].axhline(0, ls='-', linewidth = 1, color = 'k')
    axes[axes_indx].axvline(0, ls='-', linewidth = 1, color = 'k')
    axes[axes_indx].set_ylim(45, 25)
    axes[axes_indx].set_xlim(-10, 10)

    # plot the mean intensity if an image
    axes_indx = 6
    sns.kdeplot(data = meta_rms_df, x = 'intensity_mean', hue = "RMS per Image", ax = axes[axes_indx])
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Sensitivity to the mean intensity')
    axes[axes_indx].set_xlabel('Mean intensity per image')
    
    # plot the std dev of the intensity of an image
    axes_indx = 7
    sns.kdeplot(data = meta_rms_df, x = 'intensity_std', hue = "RMS per Image", ax = axes[axes_indx])
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Sensitivity to the std deviation of the intensity')
    axes[axes_indx].set_xlabel('Standard deviation of intensity per image')
    
    # prepare for the sensitivity analysis
    # prep the data for the random forest:
    meta_x = meta_rms_df[['angle_test', 'd_Leye_Reye', 'head_turn', 
                      'mid_x', 'mid_y',
                      'intensity_mean', 'intensity_std']]
    meta_x.columns = ['Tilt', 'Warping', 'Head Turn', 
                      'Translation Horizontal', 'Translation Vertical',
                     'Mean Intensity', 'Std Dev Intensity']
    meta_y = meta_rms_df['rms']
    
    # initialize the random forest to assess variable sensistivity
    rnd_reg = RandomForestRegressor()
    rnd_reg.fit(meta_x, meta_y)

    feature_importance = pd.DataFrame({'feature': meta_x.columns, 'importance': rnd_reg.feature_importances_})

    # plot the sensitivity to each meta variable:
    axes_indx = 1
    aa = feature_importance.sort_values(by = 'importance', ascending=False)
    sns.barplot(data = aa, y = 'feature', x ='importance', color = 'b', ax = axes[axes_indx])
    axes[axes_indx].set_title('Fig. ' + str(axes_indx+1) + ': Meta feature importance')
    axes[axes_indx].set_xlabel('Feature importance')
    axes[axes_indx].set_ylabel('')

    plt.show()

def error_analysis(X_test, y_test, y_pred, facial_feature_vec, plot_results = False, use_predicted = False):
    '''This is the main function for running error analysis.'''
    
    # calculate the RMSE in pixels for all the images (per feature)

    rmse_df = calc_rmse(y_test, y_pred, facial_feature_vec)

    summary_rmse = pd.concat([rmse_df.mean(axis = 0), rmse_df.median(axis = 0)], axis = 1, ignore_index = True)
    summary_rmse.columns = ['mean', 'median']

    # build a data frame that collects all the meta calculations
    
    y_test_df = pd.DataFrame(y_test, columns = facial_feature_vec)
    y_pred_df = pd.DataFrame(y_pred, columns = facial_feature_vec)
    test_minus_pred_df = y_test_df - y_pred_df
    
    meta_rms_df = pd.DataFrame(rmse_df.mean(axis = 1), columns = ['rms'])
    meta_rms_df['angle_test'] = y_test_df.apply(calc_angle, axis = 'columns')
    meta_rms_df['angle_pred'] = y_pred_df.apply(calc_angle, axis = 'columns')
    
    # do we want to use the predicted x,y coords to calculate the meta features?
    
    if use_predicted:
        meta_rms_df[['d_Leye_Reye', 
                     'd_Leye_nose',  'd_Reye_nose', 
                     'd_Leye_mouth', 'd_Reye_mouth']] = y_pred_df.apply(calc_distances, axis = 'columns', result_type = 'expand')
        meta_rms_df[['mid_x', 'mid_y']] = y_pred_df.apply(calc_mid_point_between_eyes, axis = 'columns', result_type = 'expand')
    else:
        meta_rms_df[['d_Leye_Reye', 
                     'd_Leye_nose',  'd_Reye_nose', 
                     'd_Leye_mouth', 'd_Reye_mouth']] = y_test_df.apply(calc_distances, axis = 'columns', result_type = 'expand')
        meta_rms_df[['mid_x', 'mid_y']] = y_test_df.apply(calc_mid_point_between_eyes, axis = 'columns', result_type = 'expand')

    
    
    meta_rms_df[['delta_left_eye_center_x', 'delta_left_eye_center_y',
                 'delta_right_eye_center_x', 'delta_right_eye_center_y']]  = 48 * test_minus_pred_df[['left_eye_center_x', 'left_eye_center_y',
                                                                                                       'right_eye_center_x', 'right_eye_center_y']]
    #meta_rms_df['ratio_eyes_nose'] = meta_rms_df['d_Leye_nose'] / meta_rms_df['d_Reye_nose']
    meta_rms_df['head_turn'] = meta_rms_df['d_Leye_nose'] - meta_rms_df['d_Reye_nose']

    
    # if we're not plotting, we don't need this column:
    if plot_results:
        meta_rms_df["rms_state"] = pd.qcut(meta_rms_df['rms'], q = 2, labels = ["low", "high"])
        #meta_rms_df["rms_state"] = pd.qcut(meta_rms_df['rms'], q = 2)

    meta_rms_df['intensity_mean'] = X_test.mean(axis = 1)
    meta_rms_df['intensity_std'] = X_test.std(axis = 1)

    # create the summary plots:
    
    if plot_results:
        plot_error_analysis(rmse_df, meta_rms_df)
    
    return(summary_rmse, meta_rms_df)


