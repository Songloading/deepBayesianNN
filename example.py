import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import pyro
from pyro.distributions import Normal
from pyro.distributions import Categorical
from pyro.optim import Adam
from pyro.infer import SVI
from pyro.infer import Trace_ELBO

import matplotlib.pyplot as plt
import seaborn as sns

class BNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output
        
def model(x_data, y_data, net):
    # define prior destributions
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight),
                                      scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias),
                                      scale=torch.ones_like(net.fc1.bias))
    outw_prior = Normal(loc=torch.zeros_like(net.out.weight),
                                       scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias),
                                       scale=torch.ones_like(net.out.bias))
    
    priors = {
        'fc1.weight': fc1w_prior,
        'fc1.bias': fc1b_prior, 
        'out.weight': outw_prior,
        'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    lifted_reg_model = lifted_module()
    
    lhat = F.log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)
