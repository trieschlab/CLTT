#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import math
import torch

# configuration module
# -----
import config


# custom functions
# -----

def update_target_net(net, net_target, tau=config.TAU):
    """
    function used to update the target net parameters to follow the running exponential average of online network.
        net: online network
        net_target: target network
        tau: hyper-parameter that controls the update
    """
    for param, target_param in zip(net.parameters(), net_target.parameters()):
        target_param.data.copy_((1 - tau) * param.data + tau * target_param.data)


def save_model(net, writer, epoch):
    """
        function used to save model parameters to the log directory
            net: network to be saved
            writer: summary writer to get the log directory
            epoch: epoch indicator
    """
    log_dir = writer.get_logdir()
    path = os.path.join(log_dir, 'models')
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(net.state_dict(), os.path.join(path, f'epoch_{epoch}.pt'))


def load_model(net, path):
    """
        function used to load model parameters to the log directory
            net: network to load the parameters
            path: path to the saved model state dict
    """
    net.load_state_dict(torch.load(path))

# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
