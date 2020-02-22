# -*- coding: utf-8 -*-

'''
    auto_encoder.py
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


# AutoEncoder class
class StackedAutoEncoder(nn.Module):
    def __init__(self):
        super(StackedAutoEncoder, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

sae = StackedAutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)


# Training
nb_epochs = 200

for epoch in range(1, nb_epochs + 1):
    train_loss = list()
    
    for id_user in range(nb_users):
        inp = Variable(training_set[id_user]).unsqueeze(0)
        # inp = torch.FloatTensor([training_set[id_user].tolist()])
        target = inp.clone()
        
        if torch.sum(target.data > 0) > 0:
            output = sae(inp)
            target.require_grad = False
            output[target == 0] = 0
            
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss.append(np.sqrt(loss.data.item() * mean_corrector))
            
            optimizer.step()
    
    print('epoch {}'.format(epoch), end=' ')
    print('loss {:.4f}'.format(sum(train_loss)/len(train_loss)))
            
del epoch, id_user, train_loss, inp, loss


# Testing
test_loss = list()

for id_user in range(nb_users):
    inp = torch.FloatTensor([training_set[id_user].tolist()])
    target = torch.FloatTensor([test_set[id_user].tolist()])
    
    if torch.sum(target.data > 0) > 0:
        output = sae(inp)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss.append(np.sqrt(loss.data.item() * mean_corrector))
        
print('Test loss {:.4f}'.format(sum(test_loss)/len(test_loss)))
