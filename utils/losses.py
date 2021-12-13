#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import math

# configuration module
# -----
import config

# utilities
# -----
import torch.nn.functional as F
import torch.nn as nn
import torch


# custom classes
# -----

class RELIC_TT_Loss(nn.Module):
    """
        RELIC loss which minimizes similarities the same between the anchor and different views of other samples,
            i.e. x and its pair x_pair
    """

    def __init__(self, sim_func):
        """Initialize the RELIC_TT_Loss class"""
        super(RELIC_TT_Loss, self).__init__()
        self.sim_func = sim_func


    def forward(self, x, x_pair):
        """
        params:
            x: representation tensor (Tensor)
            x_pair: tensor of the same size as x which should be the pair of x (Tensor)
        return:
            loss: the loss of RELIC-TT (Tensor)
        """
        sim = self.sim_func(x, x_pair)
        loss = F.kl_div(sim.softmax(-1).log(), sim.T.softmax(-1), reduction='batchmean')
        return loss


class BYOL_TT_Loss(nn.Module):
    """
        BYOL loss that maximizes cosine similarity between the online projection (x) and the target projection(x_pair)
    """

    def __init__(self, sim_func):
        """Initialize the SimCLR_TT_Loss class"""
        super(BYOL_TT_Loss, self).__init__()
        self.sim_func = sim_func

    def forward(self, x, x_target, labels=None):
        """
        params:
            x: representation tensor (Tensor)
            x_pair: tensor of the same size as x which should be the pair of x (Tensor)
        return:
            loss: the loss of BYOL-TT (Tensor)
        """
        loss = 2-2*self.sim_func(x, x_target).diag().mean()
        return loss


class Decorrelation_TT_Loss(nn.Module):
    """
        De-correlation loss that minimize every entry of Pearson's correlation matrix except the diagonal entries
    """

    def __init__(self, dim, device):
        """Initialize the SimCLR_TT_Loss class"""
        super(Decorrelation_TT_Loss, self).__init__()
        # mask for decorrelation loss that rules out the trace entries
        #self.MASK_DECOR = 1 - torch.eye(dim).to(device)

    def forward(self, x, x_pair=None):
        """
            params:
            x: representation tensor (Tensor)
        return:
            loss: De-correlation loss (Tensor)
        """
        #x = x - x.mean(0, keepdim=True)
        #pearson = (x.T @ x) / (config.BATCH_SIZE)
        #std_x = x.std(0, keepdim=True)
        #pearson_loss = pearson ** 2 / (std_x.T @ std_x) * self.MASK_DECOR
        #loss = pearson_loss.mean()
        loss = torch.triu(torch.abs(torch.corrcoef(x))).fill_diagonal_(0).mean()
        return loss


class SimCLR_TT_Loss(nn.Module):
    def __init__(self, sim_func, batch_size, temperature):
        """Initialize the SimCLR_TT_Loss class"""
        super(SimCLR_TT_Loss, self).__init__()

        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_func = sim_func

    def mask_correlated_samples(self, batch_size):
        """
        mask_correlated_samples takes the int batch_size
        and returns an np.array of size [2*batchsize, 2*batchsize]
        which masks the entries that are the same image or
        the corresponding positive contrast
        """
        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)

        # fill off-diagonals corresponding to positive samples
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, x, x_pair, labels=None):
        """
        Given a positive pair, we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        to control for negative samples we just cut off losses
        """
        N = 2 * self.batch_size

        z = torch.cat((x, x_pair), dim=0)

        sim = self.sim_func(z, z) / self.temperature

        # get the entries corresponding to the positive pairs
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        # we take all of the negative samples
        negative_samples = sim[self.mask].reshape(N, -1)

        if config.N_negative:
            # if we specify N negative samples: do random permutation of negative sample losses,
            # such that we do consider different positions in the batch
            negative_samples = torch.take_along_dim(
                negative_samples, torch.rand(*negative_samples.shape, device=config.DEVICE).argsort(dim=1), dim=1)
            # cut off array to only consider N_negative samples per positive pair
            negative_samples = negative_samples[:, :config.N_negative]
            # so what we are doing here is basically using the batch to sample N negative
            # samples.

        # the following is more or less a trick to reuse the cross-entropy function for the loss
        # Think of the loss as a multi-class problem and the label is 0
        # such that only the positive similarities are picked for the numerator
        # and everything else is picked for the denominator in cross-entropy
        labels = torch.zeros(N).to(config.DEVICE).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

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
