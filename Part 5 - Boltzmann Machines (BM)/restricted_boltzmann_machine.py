# -*- coding: utf-8 -*-

'''
    restricted_boltzmann_machine.py
'''


# Importing the libraries
from torch.autograd import Variable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data


# Importing the dataset
movies  = pd.read_csv('ml-1m/movies.dat',  sep='::', header=None, engine='python', encoding='latin-1')
users   = pd.read_csv('ml-1m/users.dat',   sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')


# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', sep='\t', header=None)
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', sep='\t', header=None)
test_set = np.array(test_set, dtype='int')


# Getting the number of users and movies
nb_users  = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = list()
    
    for i in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == i]
        id_ratings = data[:, 2][data[:, 0] == i]
        
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
    
        new_data.append(list(ratings))
        
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)
    
   
# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
    

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked) 
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

   
# Restricted Boltzmann Machine Class 
class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden):
        self.WEIGHT = torch.randn(n_hidden, n_visible)
        self.b_hidden = torch.randn(1, n_hidden)
        self.b_visible = torch.randn(1, n_visible)
    
    def sample_hidden(self, visible):
        wx = torch.mm(visible, self.WEIGHT.t())
        activation = wx + self.b_hidden.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_visible(self, hidden):
        wy = torch.mm(hidden, self.WEIGHT)
        activation = wy + self.b_visible.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.WEIGHT += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b_visible += torch.sum((v0 - vk), 0)
        self.b_hidden += torch.sum((ph0 - phk), 0)


# Training
nv = len(training_set[0])
nh = 250
batch_size = 25
rbm = RestrictedBoltzmannMachine(nv, nh)

n_epoch = 20

for epoch in range(1, n_epoch + 1):
    train_loss = list()
    
    for id_user in range(0, nb_users - batch_size, batch_size):
        v0 = training_set[id_user:id_user + batch_size]
        vk = training_set[id_user:id_user + batch_size]
        
        ph0, _ = rbm.sample_hidden(v0)
        
        for k in range(10):
            _, hk = rbm.sample_hidden(vk)
            _, vk = rbm.sample_visible(hk)
            vk[v0 < 0] = v0[v0 < 0]
        
        phk, _ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        
        train_loss.append(torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])))
        
    print('epoch {}'.format(epoch), end=' | ')
    print('loss {:.4f}'.format(sum(train_loss)/len(train_loss)))

del nv, nh, epoch, train_loss, id_user, v0, vk, ph0, phk, k
 

# Testing
test_loss = list() 

for id_user in range(0, nb_users):
    v = training_set[id_user:id_user + 1]
    vt = test_set[id_user:id_user + 1]
    
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_hidden(v)
        _, v = rbm.sample_visible(h)
        test_loss.append(torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])))
        print(vt[vt >= 0], v[vt >= 0])

del id_user, v, vt, h
    
print()
print('test loss {:.4f}'.format(sum(test_loss)/len(test_loss)))
    
    
    
    
    
    
