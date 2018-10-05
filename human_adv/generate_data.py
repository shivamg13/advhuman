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

x, y = read_data("MNIST_data/train")
x_test, y_test = read_data("MNIST_data/t10k")
x, x_test = map(preprocess, [x, x_test])
n_classes = int(1 + np.max(y))

def save_image(x_save, y_save, size, location, x_prefix, data_description):
    with open(location+'/'+x_prefix+'_desc.txt', 'w') as outfile:
        outfile.write(data_description);        
    with open(location+'/'+x_prefix+'_x_dump.json', 'w') as outfile:
        json.dump(x_save.tolist(), outfile);
    with open(location+'/'+x_prefix+'_y_dump.json', 'w') as outfile:
        json.dump(y_save.tolist(), outfile);
    plt.ioff()
    
    for i in range(np.size(x_save,0)):
        img = np.reshape(x_save[i,:],(size,size))
        plt.imshow(img);
        plt.savefig(location+'/'+x_prefix+'_'+str(i)+'.png', bbox_inches='tight')
        plt.cla();
    plt.ion()
        

#Take 10000 training examples and 1000 test example

#x,y = x[0:20000,:], y[0:20000];
#x_test,y_test = x_test[0:2000,:], y_test[0:2000];

inds = (y==0) + (y==1) + (y==5);
inds_test = (y_test == 0) + (y_test == 1) + (y_test ==5);

x,y = x[inds,:], y[inds];
x_test,y_test = x_test[inds_test,:], y_test[inds_test];
#y[y==2]=1;
#y_test[y_test==2]=1;
y[y==5]=2;
y_test[y_test==5]=2;

x = x[0:18000,:];
y = y[0:18000];
x_test = x_test[0:3000,:];
y_test = y_test[0:3000];

print(x.shape, y.shape)

num_train = x.shape[0];
num_test = x_test.shape[0];

#%%
isize = 225;
np.random.seed(1);
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
#    for j in range(9):
#        xf = 3*(math.floor(j/3));
#        yf = 3*(j%3);
#        xs = 3*(math.floor(cur_perm[j]/3));
#        ys = 3*(cur_perm[j]%3);
#        #print((xf,yf),(xs,ys));
#        img2_scaled[xf:xf+3,yf:yf+3]=img1_scaled[xs:xs+3,ys:ys+3];
#    x_rot[i,:] = img2_scaled.reshape(isize)
        
#    for j in range(3):
#        for k in range(3):
#            np.random.shuffle(img1_scaled[5*j:5*(j+1),5*k:5*(k+1)])
#            #np.random.shuffle(img1_scaled[5*j:5*(j+1),5*k:5*(k+1)].T)
#    x_rot[i,:] = img1_scaled.reshape(isize)
    #imshow(rescale(img1, 14/22),14)
    #x_cropped[i,:] = rescale(img1, 14/22).reshape(196)


x_cropped_test = np.zeros((np.size(x_test,0),isize));
x_smooth_test = np.zeros((np.size(x_test,0),isize));
for i in range(np.size(x_test,0)):
    img1 = crop_center(x_test[i,:].reshape(28,28), 3, 3)
    #img1_scaled = img1
    img1_scaled = rescale(img1, 15/22)
    img2_scaled = np.clip(2*gaussian_filter(img1_scaled,0.5),0,1);
    x_cropped_test[i,:] = img1_scaled.reshape(isize)
    x_smooth_test[i,:] = img2_scaled.reshape(isize)
#    for j in range(9):
#        xf = 3*(math.floor(j/3));
#        yf = 3*(j%3);
#        xs = 3*(math.floor(cur_perm[j]/3));
#        ys = 3*(cur_perm[j]%3);
#        #print((xf,yf),(xs,ys));
#        img2_scaled[xf:xf+3,yf:yf+3]=img1_scaled[xs:xs+3,ys:ys+3];
#    x_rot_test[i,:] = img2_scaled.reshape(isize)


#%%

dim = 784;
np.random.seed(0);
A = 2*(np.random.binomial(1,0.5,size=(dim,dim))-0.5)/sqrt(dim);
x_rot = np.clip(np.dot(x,A),-0.5,0.5)+0.5;
x_rot_test = np.clip(np.dot(x_test,A),-0.5,0.5)+0.5;

#%%

dim = 225;
np.random.seed(0);
A = np.random.permutation(np.eye(dim));
x_rot = np.dot(x_smooth,A);
x_rot_test = np.dot(x_smooth_test,A);
#%%
x_rot = x_smooth;
x_rot_test = x_smooth_test;

#%%
print('training svm model...')        

#Choice of C makes a big difference sometimes
model = svm.LinearSVC(max_iter = 5e3,C=0.001)

model.fit(x_rot,y)
y_test_pred = model.predict(x_rot_test)
y_train_pred = model.predict(x_rot);
print('Train error = ', np.mean(y_train_pred!=y))
print('Test error = ', np.mean(y_test_pred!=y_test))

#%%

coefs = model.coef_;
coefs = coefs[0 , :];
inds = y_test!=0;
x_source = x_rot_test[inds,:];
x_adv = np.clip(x_source + np.sign(coefs)*0.2,0,1);
y_source = y_test[inds];

y_source_pred = model.predict(x_adv);
print('Adv error = ', np.mean(y_source!=y_source_pred))
co=0;

#%%
np.random.seed(0);
coefs = model.coef_;
eps = 0.2;
train_set = x_smooth[0:1000,:];
test_set = x_smooth_test[0:1000,:];
y_train_set = y[0:1000];
y_test_set = y_test[0:1000];

rot_train_set = x_rot[0:1000,:];
rot_test_set = x_rot_test[0:1000,:];

rot_noise = np.random.randint(0,2,np.shape(rot_test_set))*eps*2 - eps;
rot_test_set_noisy = np.clip(rot_test_set + rot_noise,0,1);

y_adv_target = np.zeros(1000);
rot_test_set_adv = np.zeros(np.shape(rot_test_set));

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
    
    rot_test_set_adv[i,:] = np.clip(rot_test_set[i,:] + np.sign(coefs[target_label,:])*eps,0,1);
    
test_set_noisy = np.dot(rot_test_set_noisy ,A.T);
test_set_adv = np.dot(rot_test_set_adv ,A.T);

#%%

save_image(train_set, y_train_set, 15, 'imgtest/normal/train', 'train', 'Training data, normal images, 15 X 15');
print('Done');
save_image(test_set, y_test_set, 15, 'imgtest/normal/test', 'test', 'Test data, normal images, 15 X 15');
print('Done');
save_image(test_set_noisy, y_test_set, 15, 'imgtest/normal/test_noisy', 'test_noisy', 'Test data with +-0.2 noise, normal images, 15 X 15');
print('Done');
save_image(test_set_adv, y_adv_target, 15, 'imgtest/normal/test_adv', 'test_adv', 'Test data with linf 0.2 adversarial noise, normal images, 15 X 15, y dump contains target y label');
print('Done');

save_image(rot_train_set, y_train_set, 15, 'imgtest/rotated/train', 'train', 'Training data, rotated images, 15 X 15');
print('Done');
save_image(rot_test_set, y_test_set, 15, 'imgtest/rotated/test', 'test', 'Test data, rotated images, 15 X 15');
print('Done');
save_image(rot_test_set_noisy, y_test_set, 15, 'imgtest/rotated/test_noisy', 'test_noisy', 'Test data with +-0.2 noise, rotated images, 15 X 15');
print('Done');
save_image(rot_test_set_adv, y_adv_target, 15, 'imgtest/rotated/test_adv', 'test_adv', 'Test data with linf 0.2 adversarial noise, rotated images, 15 X 15, y dump contains target y label');
print('Done');




#%%
coefs = model.coef_;

inp = input("Stop('stop'),Train('tr'),Test('te'):")


tot = 0;
cor = 0;

ind_vec = np.zeros((400,1));
human_perf = np.zeros((400,4));#true label, guessed label, adv or not, target label if adv
co = 0;

while(inp!='stop'):
    if(inp=='tr'):
        req1 = int(input("Which label (0,1,2)?"));
        req2 = int(input("Which label (0,1,2)?"));

        
        ind1 = np.random.randint(0,1700);
        ind2 = np.random.randint(0,1700);
        
        img1 = x_rot[y==req1,:][ind1,:];
        img2 = x_rot[y==req2,:][ind2,:];
        #img4 = np.append([img1, img2, img3], axis=0);
        show_images([img1,img2],1,[req1,req2]);
        #imshow(x_rot[y==req,:][ind,:]);
    else:
        ind1 = np.random.randint(0,800);
        ind2 = np.random.randint(0,800);
        ind_vec[co] = ind1;
        label = np.random.randint(0,3);
        human_perf[co,0] = label;
        human_perf[co,3] = 4;
        human_perf[co,2] = 0;
        img1 = x_rot_test[y_test==label,:][ind1,:];
        img2 = x_rot_test[y_test==label,:][ind2,:];
        #img4 = np.append([img1, img2, img3], axis=0);te
        
        cur_rand = np.random.rand();
        eps = 0.2;
        #print(cur_rand)
        if(cur_rand<1):
            #print('adv')
            #show adversarial image
            human_perf[co,2]=1;
            if label==0:
                target_label = np.random.randint(1,3)
                
            elif label == 1:
                target_label = np.random.randint(0,2)*2;
            elif label==2:
                target_label = np.random.randint(0,2);
            else:
                print("error! undefined label.")
            #print('before:',human_perf[co,3])
            human_perf[co,3] = target_label;
            
#            if target_label==0:
#                img1 = x_adv_0[y_test==label,:][ind1,:];
#            elif target_label==1:
#                img1 = x_adv_1[y_test==label,:][ind1,:];
#            elif target_label==2:
#                img1 = x_adv_2[y_test==label,:][ind1,:];
#            else:
#                print("error! undefined target label.")
            
            
            #print('after:',human_perf[co,3])
            #img1 = np.clip(img1 + (3*coefs[target_label,:])/np.linalg.norm(coefs[target_label,:]),0,1);
            img1 = np.clip(img1 + np.sign(coefs[target_label,:])*eps,0,1);
        elif(cur_rand > 2/3):
            #print('rand')
            #Show image with random noise  
            img1_noise = np.random.randint(0,2,np.size(img1))*eps*2 - eps;
            img1 = np.clip(img1 + img1_noise,0,1);
            human_perf[co,2]=2;
            
        show_images([img1],1);
        

        
        human_perf[co,1] = int(input("Label image:"));
        
        tot = tot+1;
        cor = cor + np.sum(human_perf[co,1]==label);
        print(human_perf[co,1]==label);
        co = co + 1;
        
    inp = input("Stop('stop'),Train('tr'),Test('te'):")        
        
        
        #Write code to randomly show adversarial or nice data, and record the labels suggested by humans
        
#%%

adv_ind = (human_perf[0:co,2]==1);
rand_ind = (human_perf[0:co,2]==2);
normal_ind = (human_perf[0:co,2]==0);
human_perf1 = human_perf[0:co,:];

human_adv = human_perf1[adv_ind,:];
human_normal = human_perf1[normal_ind,:];
human_rand = human_perf1[rand_ind,:];

np.mean(human_normal[:,0]==human_normal[:,1])

np.mean(human_adv[:,0]==human_adv[:,1])

np.mean(human_rand[:,0]==human_rand[:,1])

np.mean(human_adv[:,1]==human_adv[:,3])

conf_mat_normal = np.zeros((3,3));
conf_mat_rand = np.zeros((3,3));
conf_mat_adv = np.zeros((3,3));
conf_mat_adv_target = np.zeros((3,3));
conf_mat_adv_target_1 = np.zeros((3,3));

for i in range(np.size(human_normal,0)):
    conf_mat_normal[int(human_normal[i,0]),int(human_normal[i,1])]+=1;
    
for i in range(np.size(human_adv,0)):
    conf_mat_adv[int(human_adv[i,0]),int(human_adv[i,1])]+=1;

for i in range(np.size(human_rand,0)):
    conf_mat_rand[int(human_rand[i,0]),int(human_rand[i,1])]+=1;
    
for i in range(np.size(human_adv,0)):
    conf_mat_adv_target[int(human_adv[i,3]),int(human_adv[i,1])]+=1;
    
for i in range(np.size(human_adv,0)):
    if human_adv[i,0]!=human_adv[i,1]:
        conf_mat_adv_target_1[int(human_adv[i,3]),int(human_adv[i,1])]+=1;

#%%

import pickle

f = open('scale15_permute_lin2_strong_attack.pckl', 'wb')
pickle.dump([ind_vec,human_adv,human_normal,human_perf, human_rand, conf_mat_normal, conf_mat_rand, conf_mat_adv, conf_mat_adv_target], f)
f.close()

f = open('store.pckl', 'rb')
obj = pickle.load(f)
f.close()

#%%
for i in range(np.size(human_perf,0)):
    if human_perf[i,2]==1 and  human_perf[i,0]!=human_perf[i,1] and i!=5 and i!=8 and i!=45 and i!=51 and i!=60 and i!=90 and i!=94 and i!=119 and i!=140 and i!=145 and i!=155 and i!=160:
        print(i)
        if human_perf[i,3]==0:
            print(0)
            imshow(np.dot(x_adv_0[y_test==human_perf[i,0],:][int(ind_vec[i]),:], np.transpose(A)),15);
        elif human_perf[i,3]==1:
            print(1)
            imshow(np.dot(x_adv_1[y_test==human_perf[i,0],:][int(ind_vec[i]),:], np.transpose(A)),15);
        elif human_perf[i,3]==2:
            print(2)
            imshow(np.dot(x_adv_2[y_test==human_perf[i,0],:][int(ind_vec[i]),:], np.transpose(A)),15);
        break;