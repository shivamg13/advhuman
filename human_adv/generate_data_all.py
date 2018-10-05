import numpy as np
from math import sqrt
from scipy import stats
import os
from scipy import stats
import struct
import itertools
import time
import matplotlib
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import eigh
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import scipy as sp
import math
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize
from skimage.transform import rescale
import png
import imageio
import json

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(np.reshape(image, (15, 15)))
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    plt.pause(0.001)

def crop_center(img,cropy, cropx):
    y,x = img.shape
    return img[cropy:y-cropy,cropx:x-cropx]    

def imshow(img, size):
  plt.cla()
  plt.imshow(np.reshape(img, (size, size)))
  plt.pause(0.001)
  plt.savefig("image.png")

def read_data(fname_root):
  fname_img = fname_root + "-images-idx3-ubyte"
  fname_lbl = fname_root + "-labels-idx1-ubyte"
  with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
  with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows * cols)
  return img.astype(np.float32), lbl.astype(int)

def preprocess(x): return x/255

def save_image(x_save, y_save, size, location, x_prefix, data_description):
    if not os.path.exists(location):
        os.makedirs(location)        
    with open(location+'/'+x_prefix+'_desc.txt', 'w') as outfile:
        outfile.write(data_description);        
    with open(location+'/'+x_prefix+'_x_dump.json', 'w') as outfile:
        json.dump(x_save.tolist(), outfile);
    with open(location+'/'+x_prefix+'_y_dump.json', 'w') as outfile:
        json.dump(y_save.tolist(), outfile);
    plt.ioff()
    
    for i in range(np.size(x_save,0)):
        img = np.reshape(x_save[i,:],(size,size))
        plt.imshow(img, cmap='gray_r');
        plt.savefig(location+'/'+x_prefix+'_'+str(i)+'.png', bbox_inches='tight')
        plt.cla();
    plt.ion()
    
    

    
x_all, y_all = read_data("MNIST_data/train")
x_test_all, y_test_all = read_data("MNIST_data/t10k")
x_all, x_test_all = map(preprocess, [x_all, x_test_all])
n_classes = int(1 + np.max(y_all))

triplets = np.array([[0,1,5],[2,8,9],[3,6,7],[3,4,5]]);

for triplet in triplets:
    print(triplet)
    inds = (y_all==triplet[0]) + (y_all==triplet[1]) + (y_all==triplet[2]);
    inds_test = (y_test_all == triplet[0]) + (y_test_all == triplet[1]) + (y_test_all == triplet[2]);
    
    x,y = x_all[inds,:], y_all[inds];
    x_test,y_test = x_test_all[inds_test,:], y_test_all[inds_test];
    
    y[y==triplet[0]]=10;
    y[y==triplet[1]]=11;
    y[y==triplet[2]]=12;
    
    y[y==10] = 0;
    y[y==11] = 1;
    y[y==12] = 2;
    
    y_test[y_test==triplet[0]]=10;
    y_test[y_test==triplet[1]]=11;
    y_test[y_test==triplet[2]]=12;
    
    y_test[y_test==10] = 0;
    y_test[y_test==11] = 1;
    y_test[y_test==12] = 2;
    
    #y[y==2]=1;
    #y_test[y_test==2]=1;
    #y[y==5]=2;
    #y_test[y_test==5]=2;
        
    print(x.shape, y.shape)
    
    num_train = x.shape[0];
    num_test = x_test.shape[0];
    
    isize = 225;

    x_cropped = np.zeros((np.size(x,0),isize));
    x_smooth = np.zeros((np.size(x,0),isize));
    cur_perm = np.random.permutation(9)
    for i in range(np.size(x,0)):
        img1 = crop_center(x[i,:].reshape(28,28), 3, 3)
        #img1_scaled = img1
        img1_scaled = rescale(img1, 15/22)
        img2_scaled = np.clip(2*gaussian_filter(img1_scaled,0.5),0,1);
        x_cropped[i,:] = img1_scaled.reshape(isize)
        x_smooth[i,:] = img2_scaled.reshape(isize)
        
    x_cropped_test = np.zeros((np.size(x_test,0),isize));
    x_smooth_test = np.zeros((np.size(x_test,0),isize));
    for i in range(np.size(x_test,0)):
        img1 = crop_center(x_test[i,:].reshape(28,28), 3, 3)
        #img1_scaled = img1
        img1_scaled = rescale(img1, 15/22)
        img2_scaled = np.clip(2*gaussian_filter(img1_scaled,0.5),0,1);
        x_cropped_test[i,:] = img1_scaled.reshape(isize)
        x_smooth_test[i,:] = img2_scaled.reshape(isize)
    
    for random_seed in [0,1,2,3,4]:
        dim = 225;
        np.random.seed(random_seed);
        A = np.random.permutation(np.eye(dim));
        x_rot = np.dot(x_smooth,A);
        x_rot_test = np.dot(x_smooth_test,A);
        
        print('training svm model...')        

        #Choice of C makes a big difference sometimes
        model = svm.LinearSVC(max_iter = 5e3,C=0.001)
        
        model.fit(x_smooth,y)
        y_test_pred = model.predict(x_smooth_test)
        y_train_pred = model.predict(x_smooth);
        print('Train error = ', np.mean(y_train_pred!=y))
        print('Test error = ', np.mean(y_test_pred!=y_test))
        
        coefs = model.coef_;
        eps = 0.2;
        train_set = x_smooth[0:1000,:];
        test_set = x_smooth_test[0:1000,:];
        y_train_set = y[0:1000];
        y_test_set = y_test[0:1000];
        
        rot_train_set = x_rot[0:1000,:];
        rot_test_set = x_rot_test[0:1000,:];
        
        normal_noise = np.random.randint(0,2,np.shape(rot_test_set))*eps*2 - eps;
        test_set_noisy = np.clip(test_set + normal_noise,0,1);
        rot_test_set_noisy = np.dot(test_set_noisy ,A);
        
        y_adv_target = np.zeros(1000);
        test_set_adv = np.zeros(np.shape(test_set));
        
        for i in range(1000):
            label = y_test_set[i];
            if label==0:
                target_label = np.random.randint(1,3)                
            elif label == 1:
                target_label = np.random.randint(0,2)*2;
            elif label==2:
                target_label = np.random.randint(0,2);
            else:
                print("error! undefined label.")
            #print('before:',human_perf[co,3])
            y_adv_target[i] = target_label;
            
            test_set_adv[i,:] = np.clip(test_set[i,:] + np.sign(coefs[target_label,:])*eps,0,1);
            
        rot_test_set_adv = np.dot(test_set_adv ,A);
        
        parent_dir = 'images_smooth/labels_'+str(triplet[0])+str(triplet[1])+str(triplet[2])+'/perm'+str(random_seed);
        
        
        save_image(train_set, y_train_set, 15, parent_dir+'/normal/train', 'train', 'Training data, normal images, 15 X 15');
        print('Done');
        save_image(test_set, y_test_set, 15, parent_dir+'/normal/test', 'test', 'Test data, normal images, 15 X 15');
        print('Done');
        save_image(test_set_noisy, y_test_set, 15, parent_dir+'/normal/test_noisy', 'test_noisy', 'Test data with +-0.2 noise, normal images, 15 X 15');
        print('Done');
        save_image(test_set_adv, y_adv_target, 15, parent_dir+'/normal/test_adv', 'test_adv', 'Test data with linf 0.2 adversarial noise, normal images, 15 X 15, y dump contains target y label');
        print('Done');
        
        save_image(rot_train_set, y_train_set, 15, parent_dir+'/rotated/train', 'train', 'Training data, rotated images, 15 X 15');
        print('Done');
        save_image(rot_test_set, y_test_set, 15, parent_dir+'/rotated/test', 'test', 'Test data, rotated images, 15 X 15');
        print('Done');
        save_image(rot_test_set_noisy, y_test_set, 15, parent_dir+'/rotated/test_noisy', 'test_noisy', 'Test data with +-0.2 noise, rotated images, 15 X 15');
        print('Done');
        save_image(rot_test_set_adv, y_adv_target, 15, parent_dir+'/rotated/test_adv', 'test_adv', 'Test data with linf 0.2 adversarial noise, rotated images, 15 X 15, y dump contains target y label');
        print('Done');
        
