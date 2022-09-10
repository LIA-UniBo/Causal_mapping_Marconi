from Networks.TCDF_net import *
from torch import optim
from tqdm import trange
import torch.nn as nn
import numpy as np
import random
import math

class TCDFModule():
    def __init__(self,in_channels,levels,kernel_size,dilation,device,lr,epochs,confidence_s=0.8):
        self.levels = levels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.dilation = dilation
        self.device = device
        self.network = ADDSTCN(in_channels=in_channels, levels=self.levels, kernel_size=kernel_size,
                          dilation=self.dilation, device=device).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.epochs = epochs
        self.losses_train = []
        self.losses_val = []
        self.confidence_s = confidence_s

    def train_old(self,x_train,y_train,x_val=None,y_val=None,split=False):
        self.x_train = x_train
        self.y_train = y_train
        self.y_val = y_val
        self.x_val = x_val
        with trange(self.epochs) as pbar:
            for ep in pbar:
                self.optimizer.zero_grad()
                self.network.train()
                prediction = self.network(self.x_train)
                loss_training = self.loss(prediction, self.y_train)
                self.losses_train.append(loss_training.item())
                loss_training.backward()
                self.optimizer.step()
                if split:
                    self.network.eval()
                    prediction = self.network(self.x_val)
                    loss_validation = self.loss(prediction, self.y_val)
                    self.losses_val.append(loss_validation.item())
                    pbar.set_postfix(loss_training=loss_training.item(), loss_validation=loss_validation.item())
                else:
                    pbar.set_postfix(loss_training=loss_training.item())
        print("Starting loss training: " + str(self.losses_train[0]) + "    Final loss training: " + str(self.losses_train[-1]))
        if split:
            print("Starting loss val: " + str(self.losses_val[0]) + "    Final loss val: " + str(self.losses_val[-1]))


    def train(self,training,validation,split=False):
        self.training = training
        self.validation = validation
        self.losses_train = []
        with trange(self.epochs) as pbar:

            for ep in pbar:
                self.optimizer.zero_grad()
                count = 0
                self.network.train()
                loss_training = 0
                for batch_train in training:
                    for index_train in range(len(batch_train[0])):
                        x_train = batch_train[0][index_train]
                        y_train = batch_train[1][index_train].to(self.device)
                        prediction = self.network(x_train)
                        loss_training += self.loss(prediction, y_train)
                        count +=1
                loss_training.backward()
                self.optimizer.step()
                loss_training = loss_training/count
                self.losses_train.append(loss_training.item())
                if split:
                    self.network.eval()
                    loss_validation = 0
                    count = 0
                    for batch_val in validation:
                        for index_val in range(len(batch_val[0])):
                            x_val = batch_val[0][index_val]
                            y_val = batch_val[1][index_val].to(self.device)
                            prediction = self.network(x_val)
                            loss_validation += self.loss(prediction, y_val)
                            count +=1
                    loss_validation = loss_validation/count
                    self.losses_val.append(loss_validation.item())
                    pbar.set_postfix(loss_training=loss_training.item(), loss_validation=loss_validation.item())
                else:
                    pbar.set_postfix(loss_training=loss_training.item())
        print("Starting loss training: " + str(self.losses_train[0]) + "    Final loss training: " + str(self.losses_train[-1]))
        if split:
            print("Starting loss val: " + str(self.losses_val[0]) + "    Final loss val: " + str(self.losses_val[-1]))

    def compute_threshold(self):
        gaps = []
        attention_scores = self.network.attention.detach().numpy().flatten()            #scores before softmax
        attention_scores = np.sort(attention_scores)[::-1]
        if len(attention_scores)<=5:
            for index in range(len(attention_scores)):
                if attention_scores[index] < 1:
                    attention_scores = attention_scores[:index+1]
                    break
        else:
            attention_scores = attention_scores[:math.ceil(len(attention_scores)/2)]
            for index in range(len(attention_scores)):
                if attention_scores[index] < 1:
                    attention_scores = attention_scores[:index+1]
                    break
        for index in range(len(attention_scores)-1):
            gaps.append(attention_scores[index] - attention_scores[index+1])
        index = np.argmax(gaps)
        if index == 0 and attention_scores[index+1] >= 1:
            index = np.argmax(gaps[(index+1):])+1
        threshold = attention_scores[index]
        return threshold

    def compute_scores(self):
        scores = []
        potential_causes = []
        threshold = self.compute_threshold()
        attention_scores = self.network.attention.detach().numpy().flatten()
        for attention in attention_scores:
            if attention <  threshold:
                score = 0
            else:
                score = 1
            scores.append(score)
        for index in range(len(scores)):
            if scores[index]>0:
                potential_causes.append(index)
        print("Potential causes: "+str(potential_causes))
        return scores

    def PIVM(self,scores):
        self.network.eval()
        for index in range(len(scores)):
            if scores[index]==1:
                loss_permuted = 0
                count = 0
                for batch_train in self.training:
                    for index_train in range(len(batch_train[0])):
                        x_train = batch_train[0][index_train]
                        y_train = batch_train[1][index_train].to(self.device)
                        permuted = x_train[:, index,:].flatten().detach().numpy()
                        random.shuffle(permuted)
                        prediction = self.network(x_train)
                        loss_permuted += self.loss(prediction,y_train)
                        count +=1
                loss_permuted = loss_permuted / count
                diff_loss_train = self.losses_train[0] - self.losses_train[-1]
                diff_loss_permuted =  self.losses_train[0] - loss_permuted
                if diff_loss_permuted > diff_loss_train*self.confidence_s:
                    scores[index]=0
        return scores

    def PIVM_old(self, scores):
        self.network.eval()
        for index in range(len(scores)):
            if scores[index] == 1:
                x_permuted = self.x_train.clone().detach().numpy()
                permuted = x_permuted[:, index, :].flatten()
                random.shuffle(permuted)
                x_permuted[:, index, :] = permuted
                prediction = self.network(torch.Tensor(x_permuted))
                loss_permuted = self.loss(prediction, self.y_train)
                diff_loss_train = self.losses_train[0] - self.losses_train[-1]
                diff_loss_permuted = self.losses_train[0] - loss_permuted
                if diff_loss_permuted > diff_loss_train * self.confidence_s:
                    scores[index] = 0
        return scores

    def find_causes(self):
        causes = []
        scores = self.compute_scores()
        scores = self.PIVM(scores)
        for index in range(len(scores)):
            if scores[index]==1:
                causes.append(index)
        return causes

    def find_delay(self,causes):
        windows = []
        for cause in causes:
            list_weight = []
            list_weight.append(self.network.first_layer.conv1d.weight.abs().detach().numpy().reshape(self.in_channels,self.kernel_size)[cause])
            for layer in self.network.middle_layers:
                list_weight.append(layer.conv1d.weight.abs().detach().numpy().reshape(self.in_channels,self.kernel_size)[cause])
            list_weight.append(self.network.final_layer.conv1d.weight.abs().detach().numpy().reshape(self.in_channels,self.kernel_size)[cause])
            list_weight.reverse()
            weights = np.concatenate(list_weight, axis=0).reshape(self.levels,self.kernel_size)
            window = 0
            for level in range(self.levels):
                index = (self.kernel_size - 1) - np.argmax(weights[level])
                window = window + self.dilation**(self.levels - 1 - level)*index
            windows.append(window)
        return windows



