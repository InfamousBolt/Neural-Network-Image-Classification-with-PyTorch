# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256


        # TODO Define the network architecture (layers) based on these specifications.
        h, w= 31, 31
        out_channels1, out_channels2  = 16, 32
        h_after_pool1, w_after_pool1 = h // 2, w // 2
        h_final, w_final = h_after_pool1 // 2, w_after_pool1 // 2
        self.fc_input_size = out_channels2 * h_final * w_final
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            #2nd CNN
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, out_size)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
    
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, 31, 31)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        x = self.fc_layers(x)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
    
        # Important, detach and move to cpu before converting to numpy and then to python float.
        # Or just use .item() to convert to python float. It will automatically detach and move to cpu.
        self.optimizer.zero_grad()
        output = self(x)        
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    # Important, don't forget to detach losses and model predictions and convert them to the right return types.
    # Data standardization
    losses = []
    best_loss = float('inf')
    std = torch.std(train_set, dim=0)
    mean = torch.mean(train_set, dim=0)
    epsilion = 1e-8
    dev_set = (dev_set - mean) / (std + epsilion)
    train_set = (train_set - mean) / (std + epsilion)
    in_size = train_set.shape[1]
    out_size = 4
    model = NeuralNet(lrate=0.001, loss_fn=nn.CrossEntropyLoss(), in_size=in_size, out_size=out_size)
    
    
    dataset = get_dataset_from_arrays(train_set, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
   
    for i in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch in dataloader:
            x = batch['features']
            y = batch['labels']
            
            loss = model.step(x, y)
            epoch_losses.append(loss)
        
        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        model.scheduler.step(mean_loss)
        
        if mean_loss < best_loss:
            best_loss = mean_loss#better loss
            best_state = model.state_dict()
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dev_outputs = model(dev_set)
        predictions = torch.argmax(dev_outputs, dim=1).numpy()
    
    return losses, predictions, model
