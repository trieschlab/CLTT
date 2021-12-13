#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
from torch.linalg import lstsq
import torch.nn.functional as F
import pacmap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
import seaborn as sb
import numpy as np

import config


# custom functions
# -----
@torch.no_grad()
def get_representations(net, data_loader):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        net: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        representations: representations output by the network (Tensor)
        labels: labels of the original data (LongTensor)
    """
    net.eval()
    features = []
    labels = []
    for data_samples, data_labels in data_loader:
        features.append(net(data_samples.to(config.DEVICE))[0])
        labels.append(data_labels.to(config.DEVICE))

    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels


@torch.no_grad()
def lls(representations, labels, n_classes):
    """
        Calculate the linear least square prediction accuracy
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
        return:
            acc: the LLS accuracy (float)
    """
    ls = lstsq(representations, F.one_hot(labels, n_classes).type(torch.float32))
    solution = ls.solution
    acc = ((representations @ solution).argmax(dim=-1) == labels).sum() / len(representations)
    return acc


@torch.no_grad()
def wcss_bcss(representations, labels, n_classes):
    """
        Calculate the within-class and between-class average distance ratio
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
        return:
            wb: the within-class and between-class average distance ratio (float)
    """
    representations = torch.stack([representations[labels == i] for i in range(n_classes)])
    centroids = representations.mean(1, keepdim=True)
    wcss = (representations - centroids).norm(dim=-1).mean()
    bcss = F.pdist(centroids.squeeze()).mean()
    wb = wcss / bcss
    return wb


@torch.no_grad()
def get_pacmap(representations, labels, epoch, n_classes, class_labels):
    """
        Draw the PacMAP plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
        return:
            fig: the PacMAP plot (matplotlib.figure.Figure)
    """
    sb.set()
    sb.set_style("ticks")
    sb.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 2})
    color_map = get_cmap('viridis')
    legend_patches = [Patch(color=color_map(i / n_classes), label=label) for i, label in enumerate(class_labels)]
    # save the visualization result
    embedding = pacmap.PaCMAP(n_dims=2)
    X_transformed = embedding.fit_transform(representations.cpu().numpy(), init="pca")
    fig, ax = plt.subplots(1, 1)
    labels = labels.cpu().numpy()
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap=color_map, s=0.6)
    plt.title(config.MAIN_LOSS + r' $N_{fix}$=' + str(config.N_fix))
    plt.xticks([]), plt.yticks([])
    plt.legend(loc='upper left', bbox_to_anchor=(1., 1.), handles=legend_patches, fontsize=13.8)
    # ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, size=30, weight='medium')
    plt.xlabel(f'Epoch: {epoch}')
    return fig


def cosine_similarity(p_vec, q_vec):
    """
    cosine_similarity takes two numpy arrays of the same shape and returns
    a float representing the cosine similarity between two vectors
    """
    p_vec, q_vec = p_vec.flatten(), q_vec.flatten()
    return np.dot(p_vec, q_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(q_vec))


@torch.no_grad()
def get_neighbor_similarity(representations, labels, epoch, sim_func=cosine_similarity):
    """
        Draw a similarity plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
            sim_func: similarity function with two parameters
        return:
            fig: similarity plot (matplotlib.figure.Figure)
    """

    unique_labels = torch.unique(labels)

    if len(labels) != len(unique_labels):
        # calculate the mean over representations
        rep_centroid = torch.zeros([len(unique_labels), representations.shape[-1]])
        for i in range(len(unique_labels)):
            rep_centroid[i] = representations[torch.where(labels == i)[0]].mean(0)

        list_of_indices = np.arange(len(unique_labels))
        labels = list_of_indices
        representations = rep_centroid
        n_samples_per_object = 1

    else:
        list_of_indices = np.arange(len(labels))
        n_samples_per_object = 1

    distances = np.zeros([len(unique_labels), len(unique_labels)])

    # Fill a distance matrix that relates every representation of the batch
    for i in list_of_indices:
        for j in list_of_indices:
            distances[labels[i], labels[j]] += sim_func(representations[i].cpu(), representations[j].cpu())
            # distances[labels[i], labels[j]] += 1

    distances /= n_samples_per_object ** 2  # get the mean distances between representations

    # get some basic statistics
    # print('[INFO:] distance', distances.max(), distances.min(), distances.std())

    # duplicate the matrix such that you don't get to the edges when
    # gathering distances
    distances = np.hstack([distances, distances, distances])
    # plt.matshow(distances)
    # plt.show()

    # how many neighbors do you want to show (n_neighbors = n_classes for sanity check, you would have to see a global symmetry)
    n_neighbors = len(unique_labels)
    topk_dist_plus = np.zeros([len(labels), n_neighbors])
    topk_dist_minus = np.zeros([len(labels), n_neighbors])

    for k in range(n_neighbors):
        for i in range(len(unique_labels)):
            topk_dist_plus[i, k] += distances[i, i + len(unique_labels) + k]
            topk_dist_minus[i, k] += distances[i, i + len(unique_labels) - k]

    topk_dist = np.vstack([topk_dist_plus, topk_dist_minus])

    fig, ax = plt.subplots()
    ax.errorbar(np.arange(0, n_neighbors), topk_dist.mean(0), marker='.', markersize=10, xerr=None,
                yerr=topk_dist.std(0))
    ax.set_title('representation similarity')
    ax.set_xlabel('nth neighbour')
    ax.set_ylabel('cosine similarity')
    ax.set_ylim(-1.1, 1.1)
    ax.hlines(topk_dist.mean(0)[n_neighbors // 2:].mean(), -100, 100, color='gray', linestyle='--')
    ax.set_xlim(-2, n_neighbors + 2)

    return fig

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
