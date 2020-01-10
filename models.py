import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from math import floor
from torch.nn.init import xavier_uniform
import pdb

# A simple dnn model consists of three fully connected layers. 
# The input is the bow matrix of the discharge summary, and the output is a 2-dimensional vector. 
# The first dimension is the probability of no pressure ulcer, 
# and the second dimension is the probability of having PU.
class DNN_model_for_BOW(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(DNN_model_for_BOW,self).__init__()
        self.fc1 = nn.Linear(D_in,H)
        self.fc2 = nn.Linear(H,H)
        self.fc3 = nn.Linear(H,D_out)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        shortcut = x
        x = F.relu(self.fc2(x))
        x += shortcut
        x = self.dropout(x)
        logits = self.fc3(x)
        result = F.log_softmax(logits,-1).squeeze()
        return result
        
# Drop entire rows of matrix, for the ConvAttnPool below
class MyDropout(nn.Module):
    def __init__(self):
        super(MyDropout, self).__init__()
        mask = torch.zeros()
        
    def forward(self, x, pre_x,word_count):
        if self.training:
            mask = (pre_x != 0).all(0).float()
            x = x * mask
        return x

'''
The convolution model with the attention mechanism is directly pasted from:
https://github.com/jamesmullenbach/caml-mimic/blob/master/learn/models.py
Some parameters related to predicting multiple diseases are deleted.
'''   
class ConvAttnPool(nn.Module):
    def __init__(self, Y, kernel_size, num_filter_maps,embed_size=100, dropout=0.5, code_emb=None):
        super(ConvAttnPool, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(floor(kernel_size/2)))
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)
        #self.loss = F.binary_cross_entropy_with_logits()
        
    def forward(self, x, target, get_attention=True):
        #get embeddings and apply dropout
        # batch_size = B word_num=W embedding_size = H
        #sum(torch.sum(whole_word_matrix,-1) != 0)
        x = self.dropout(x)
        #B * W * H
        x = x.transpose(1, 2)
        #B * H * W
        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2))
        #B * output_channel * W * H
        #apply attention
        #U_weight = class_num * output_channel
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
        #import pdb; pdb.set_trace()
        #alpha N Y L
        #document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        #final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
       
            
        #final sigmoid to get predictions
        # Replace sigmoid + ce loss with softmax + ce_loss
        yhat = y
        proba = F.log_softmax(yhat,1)
        #pdb.set_trace()
        loss = F.binary_cross_entropy_with_logits(yhat, target)

        #loss = nn.NLLLoss(proba,target)
        return proba, loss, alpha
