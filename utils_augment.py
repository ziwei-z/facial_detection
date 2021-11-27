import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import random as rd
import importlib

import utils as ut
importlib.reload(ut)

rd.seed(19)

### mirroring ###
def hz_mirror_image(image):
    mirror_image = np.flip(image, axis=1)   # Flip column-wise (axis=1)
    return mirror_image 

vector1 = np.array([2,3,0,1,8,9,10,11,4,5,6,7,16,17,18,19,12,13,14,15,20,21,24,25,22,23,26,27,28,29])

def hz_mirror_keypoints(kpts):
    mirror_keypoints = kpts[vector1]

    for idx, value in enumerate(mirror_keypoints):
        if idx % 2 == 0: # is it an x location-coordinate
            mirror_keypoints[idx] = -1 * mirror_keypoints[idx] # flip it
        else: # if it's a y location-coordinate
            continue # do not change
    return mirror_keypoints

def mk_mirror_data(trainX, trainY):
    '''
    returns mirrored data and keypoints of trainX and trainY
    '''        
    flipped_images = np.empty_like(trainX)
    flipped_keypoints = np.empty_like(trainY)
    
    # go through TrainX and mirror the images
    for indx in range(trainX.shape[0]):
        flip_img = hz_mirror_image(trainX[indx, :, :, :])
        flipped_images[indx, :, :, :] = flip_img


    # go through TrainY and mirror the keypoint coordinates
    for indx in range(trainY.shape[0]):
        flip_kpt = hz_mirror_keypoints(trainY[indx, :])
        flipped_keypoints[indx,:] = flip_kpt
    
    #aug_trainX = np.append(trainX, flipped_images, axis = 0)
    #aug_trainY = np.append(trainY, flipped_keypoints, axis = 0)

    return flipped_images, flipped_keypoints


### Warping ###
def warp_image(image, kpts):
    '''
    changes image perspective
    input: one image and its corresponding x, y coordinates
    '''    
    x = rd.uniform(0.98, 1.02)

    rows, cols = image.shape[0:2]
    pts1 = np.float32(
        [[cols*.01*x, rows*1*x],
         [cols*.98*x, rows*1*x],
         [cols*0.04*x, rows*0.01*x],
         [cols*0.96*x, rows*0.01*x]]
    )
    pts2 = np.float32(
        [[0, rows],
         [cols, rows],
         [0, 0],
         [cols, 0]]
    )     
    
    M = cv2.getPerspectiveTransform(pts1,pts2)

    # transform image 
    warp_image = cv2.warpPerspective(image, M, (cols, rows))
    warp_image = warp_image.reshape(rows,cols,1)

    # transform label 
    warp_label = np.empty_like(kpts)
    for pt in range(kpts.shape[0]):
        warp_label[pt] = kpts[pt]
    warp_label = warp_label*48+48
    for pt in range(int(kpts.shape[0]/2)):
        x = (kpts[::2]*48+48)[pt]
        y = (kpts[1::2]*48+48)[pt]
        warp_label[::2][pt]=(M[0][0]*x + M[0][1]*y + M[0][2]) / ((M[2][0]*x + M[2][1]*y + M[2][2]))
        warp_label[1::2][pt]=(M[1][0]*x + M[1][1]*y + M[1][2]) / ((M[2][0]*x+ M[2][1]*y + M[2][2]))
    warp_label = (warp_label-48)/48
    
    return warp_image, warp_label

def mk_warp_data(trainX, trainY):
    '''
    returns warped data and keypoints of trainX and trainY
    '''
    warped_images = np.empty_like(trainX)
    warped_keypoints = np.empty_like(trainY)
    
    # go through TrainX and warp the images
    for indx in range(trainX.shape[0]):
        warp_img, warp_kpt = warp_image(trainX[indx, :, :, :], trainY[indx, :])
        warped_images[indx, :, :, :] = warp_img
        warped_keypoints[indx,:] = warp_kpt

    return warped_images, warped_keypoints

### Rotating ###
def rotate_image(image, kpts, maxrotate):
    '''
    rotates image
    input: one image and its corresponding x, y coordinates
    '''    
    x = (1-abs(np.random.normal(0, 1/3)))*maxrotate*rd.choice([-1,1])  # rotate between -10 and 10 degrees, with higher probability at bigger degrees
    
    rows, cols = image.shape[0:2]
    M = cv2.getRotationMatrix2D(((cols-1)/2,(rows-1)/2),x,1)

    # transform image 
    rotated_image = cv2.warpAffine(image, M, (cols,rows))
    rotated_image = rotated_image.reshape(rows,cols,1)
    
    # transform label 
    rotate_label = np.empty_like(kpts)
    for pt in range(kpts.shape[0]):
        rotate_label[pt] = kpts[pt]
    rotate_label = rotate_label*48+48
    for pt in range(int(kpts.shape[0]/2)):
        x = (kpts[::2]*48+48)[pt]
        y = (kpts[1::2]*48+48)[pt]
        rotate_label[::2][pt]=(M[0][0]*x + M[0][1]*y + M[0][2])
        rotate_label[1::2][pt]=(M[1][0]*x + M[1][1]*y + M[1][2])
    rotate_label = (rotate_label-48)/48
    
    return rotated_image, rotate_label

def mk_rotate_data(trainX, trainY, maxrotate):
    '''
    returns rotated data and keypoints of trainX and trainY
    '''
    rotated_images = np.empty_like(trainX)
    rotated_keypoints = np.empty_like(trainY)
    
    # go through TrainX and rotate the images
    for indx in range(trainX.shape[0]):
        rot_img, rot_kpt = rotate_image(trainX[indx, :, :, :], trainY[indx, :], maxrotate)
        rotated_images[indx, :, :, :] = rot_img
        rotated_keypoints[indx,:] = rot_kpt
    
    #aug_trainX = np.append(trainX, warped_images, axis = 0)
    #aug_trainY = np.append(trainY, warped_keypoints, axis = 0)

    return rotated_images, rotated_keypoints

### changing contrast ###
def contrast_image(image, kpts, alpha=1):
    '''
    adjusts contrast of an image using alpha
    default alpha=1 means image remains the same
    '''
    image_light = alpha*image + (1-alpha)*np.mean(image)
    # new label is the same
    new_label = np.empty_like(kpts)
    for pt in range(kpts.shape[0]):
        new_label[pt] = kpts[pt]    
    
    return image_light, new_label

def mk_contrast_data(trainX, trainY, alpha=1):
    '''
    returns rotated data and keypoints of trainX and trainY
    '''
    contrast_images = np.empty_like(trainX)
    contrast_keypoints = np.empty_like(trainY)
    
    # go through TrainX and rotate the images
    for indx in range(trainX.shape[0]):
        con_img = contrast_image(trainX[indx, :, :, :], trainY[indx, :], alpha)[0]
        con_kpt = contrast_image(trainX[indx, :, :, :], trainY[indx, :], alpha)[1]
        contrast_images[indx, :, :, :] = con_img
        contrast_keypoints[indx,:] = con_kpt

    return contrast_images, contrast_keypoints
    
### plot augmented data ###
def plot_aug(dataX, dataY, aug_dataX, aug_dataY, dataXshape=0, type=''):
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0.13,wspace=0.1,
                        left=0,right=1,bottom=0, top=1)

    Npicture = 4
    n_cols = 2
    n_rows = int(Npicture/n_cols)
    
    # randomize which pics to show
    x = round(rd.uniform(0, dataXshape-10))
    tmp_vec = np.c_[np.arange(Npicture)+x, np.arange(Npicture)+x].flatten()

    count = 1
    for irow in range(Npicture):
        #ipic = np.random.choice(X_square.shape[0])
        ipic = tmp_vec[irow]
        ax = fig.add_subplot(int(Npicture/2) , 2, count, xticks=[],yticks=[])   
        if count%2==1:
            ax.imshow(dataX[ipic].reshape(96,96),cmap="gray",vmax=1, vmin=0)
            ax.scatter(48*dataY[ipic][0::2]+ 48,48*dataY[ipic][1::2]+ 48, s = 7, marker = 's', color='red')
            ax.set_title("picture "+ str(ipic) + ' original')
        if count%2==0:
            ax.imshow(aug_dataX[ipic].reshape(96,96),cmap="gray",vmax=1, vmin=0)
            ax.scatter(48*aug_dataY[ipic][0::2]+ 48,48*aug_dataY[ipic][1::2]+ 48, s = 7, marker = 's', color='red')  
            ax.set_title("picture "+ str(ipic) + ' ' + type)
        count += 1

    plt.show()

### augment data ###
# if augmenting in at least one way 
# use this if if sum(params['augment'].values()) > 0:

def aug(params, origX, origY, maxrotate=20, contrast_alpha=0.8):
    
    trainX = origX
    trainY = origY
    
    # this ensures the left plot is always from original train X
    z = origX.shape[0]
    
    # pass through each type of augmentation
    # if true, generate data and labels and plot examples
    if params['augment']['mirror']:
        mirrorX, mirrorY = mk_mirror_data(origX, origY)
        plot_aug(origX, origY, mirrorX, mirrorY, z, type='mirrored')   
        trainX = np.append(trainX, mirrorX, axis=0)
        trainY = np.append(trainY, mirrorY, axis=0)
    if params['augment']['rotate']:
        rotateX, rotateY = mk_rotate_data(origX, origY, maxrotate)
        plot_aug(origX, origY, rotateX, rotateY, z, type='rotated')
        trainX = np.append(trainX, rotateX, axis=0)
        trainY = np.append(trainY, rotateY, axis=0)
    if params['augment']['warp']:
        warpX, warpY = mk_warp_data(origX, origY)
        plot_aug(origX, origY, warpX, warpY, z, type='warped')
        trainX = np.append(trainX, warpX, axis=0)
        trainY = np.append(trainY, warpY, axis=0)
    if params['augment']['contrast']:
        lightX, lightY = mk_contrast_data(origX, origY, contrast_alpha)
        plot_aug(origX, origY, lightX, lightY, z, type='lightened')
        trainX = np.append(trainX, lightX, axis=0)
        trainY = np.append(trainY, lightY, axis=0) 
        
    return trainX, trainY

###Creates a mask to remove pixel border####
def mask_create(border_size):
    mask_out = np.ones((96, 96, 1))
    if border_size == 1:
        mask_out[0,:] = 0
        mask_out[:,0] = 0
        mask_out[95,:] = 0
        mask_out[:,95] = 0
    else:
        i = border_size
        j = 96-i
        mask_out[:i,:] = 0
        mask_out[:,:i] = 0
        mask_out[j:96,:] = 0
        mask_out[:,j:96] = 0
    return mask_out

def apply_mask(masksize, X2):
    mask = mask_create(masksize)
    for indx in range(X2.shape[0]):
        X2[indx] = X2[indx] * mask
    return X2



