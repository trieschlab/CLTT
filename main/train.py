#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import sys

from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

# configuration module
# -----

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import config

# custom libraries
# -----
from utils.datasets import TDWDataset, MiyashitaDataset, COIL100Dataset
from utils.networks import ResNet18, MLPHead
from utils.losses import RELIC_TT_Loss, BYOL_TT_Loss, SimCLR_TT_Loss, Decorrelation_TT_Loss
from utils.general import update_target_net
from utils.evaluation import get_representations, lls, wcss_bcss, get_pacmap, get_neighbor_similarity

# similarity functions dictionary
SIMILARITY_FUNCTIONS = {
    'cosine': lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2),
    'RBF': lambda x, x_pair: -torch.cdist(x, x_pair)
}

# loss dictionary for different losses
MAIN_LOSS = {
    'SimCLR': SimCLR_TT_Loss(SIMILARITY_FUNCTIONS[config.SIMILARITY], config.BATCH_SIZE, config.TEMPERATURE),
    'BYOL': BYOL_TT_Loss(SIMILARITY_FUNCTIONS[config.SIMILARITY]),
    'supervised': lambda x, x_pair, labels: F.cross_entropy(x, labels)
}

REG_LOSS = {
    'RELIC': RELIC_TT_Loss(SIMILARITY_FUNCTIONS[config.SIMILARITY]),
    'Decorrelation': Decorrelation_TT_Loss(config.HIDDEN_DIM, config.DEVICE)
}
DATASETS = {
    'TDW': {'class': TDWDataset,
            'buffersize': 12096,
            'rgb_mean': (0.7709, 0.7642, 0.7470),
            'rgb_std': (0.0835, 0.0842, 0.0840),
        },
    'Miyashita': {'class': MiyashitaDataset,
            'buffersize': 10000,
            'rgb_mean': (0.1932, 0.2042, 0.2031),
            'rgb_std': (0.2539, 0.2616, 0.2617),
        },
    'COIL100': {'class': COIL100Dataset,
            'buffersize': 5400,
            'rgb_mean': (0.3073, 0.2593, 0.2063),
            'rgb_std': (0.2391, 0.1947, 0.1579),
        },
}
# custom function
# -----


def train():
    # prepare tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, config.RUN_NAME))
    
    # get transformations for validation and for training
    train_transform, val_transform = transforms.ToTensor(), transforms.ToTensor()
    
        
    dataset_test = DATASETS[config.DATASET]['class'](
        root='data',
        train=False,
        transform=val_transform,
        contrastive=False,
        sampling_mode=config.VIEW_SAMPLING,
        shuffle_object_order=config.SHUFFLE_OBJECT_ORDER,
        buffer_size= DATASETS[config.DATASET]['buffersize'],
        n_fix=config.N_fix)
    dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE, num_workers=0, shuffle=False)
 
    net = ResNet18().to(config.DEVICE)
    net_target = ResNet18().to(config.DEVICE).eval()

    # specific when BYOL-TT is used
    if config.MAIN_LOSS == 'BYOL':
        net_target.train()
        predictor = MLPHead(config.FEATURE_DIM, config.HIDDEN_DIM, config.FEATURE_DIM).to(config.DEVICE)
        # initialize target network
        for param_online, param_target in zip(net.parameters(), net_target.parameters()):
            param_target.data.copy_(param_online.data)  # initialize
            param_target.requires_grad = False  # not update by gradient

        optimizer = torch.optim.AdamW(list(net.parameters()) + list(predictor.parameters()), lr=config.LRATE)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.LRATE)

    # get initial result and save plot and record
    features, labels = get_representations(net, dataloader_test)
    acc = lls(features, labels, dataset_test.n_classes)
    wb = wcss_bcss(features, labels, dataset_test.n_classes)
    pacmap_plot = get_pacmap(features, labels, 0, 
        dataset_test.n_classes, dataset_test.labels)
    print(f'Initial result: Read-Out Acc:{acc * 100:>6.2f}%, WCSS/BCSS:{wb:>8.4f}')
    writer.add_scalar('accloss/accuracy', acc, 0)
    writer.add_scalar('analytics/WCSS-BCSS', wb, 0)
    writer.add_figure('PacMap', pacmap_plot, 0)
    if config.SAVE_EMBEDDING:
        writer.add_embedding(features, tag='Embedding', global_step=0)

    # decrease learning rate by a factor of 0.3 every 10 epochs
    # scheduler = StepLR(optimizer, 10, 0.3)
    if config.COSINE_DECAY:
        scheduler = CosineAnnealingLR(optimizer, T_max=config.N_EPOCHS,
            eta_min=config.LRATE * (config.LR_DECAY_RATE ** 3))
    elif config.EXP_DECAY:
        scheduler = ExponentialLR(optimizer, 1.0)
    else:
        scheduler =  StepLR(optimizer, 10, config.LR_DECAY_RATE)
    
    
    epoch_loop = tqdm(range(config.N_EPOCHS), ncols=80)
    for epoch in epoch_loop:
        epoch_loop.set_description(f"Method: {config.RUN_NAME.split('~')[0]}, Epoch: {epoch + 1}")
        dataset_train = DATASETS[config.DATASET]['class'](
            root='data',
            train=True,
            transform=train_transform,
            contrastive=True,
            sampling_mode=config.VIEW_SAMPLING,
            shuffle_object_order=config.SHUFFLE_OBJECT_ORDER,
            buffer_size=DATASETS[config.DATASET]['buffersize'],
            n_fix=config.N_fix)

        dataloader_train = DataLoader(dataset_train, batch_size=config.BATCH_SIZE,
                                      num_workers=4, shuffle=True, drop_last=True)
        if config.MAIN_LOSS != 'BYOL':
            net.train()
        training_loop = tqdm(dataloader_train)
        for (x, x_pair), labels in training_loop:
            x, y = torch.cat([x, x_pair], 0).to(config.DEVICE), labels.to(config.DEVICE)
            representation, projection = net(x)
            projection, pair = projection.split(config.BATCH_SIZE)
            if config.MAIN_LOSS == 'BYOL':
                projection = predictor(projection)
                update_target_net(net, net_target)
                with torch.no_grad():
                    pair = net_target(x)[1].split(config.BATCH_SIZE)[1]
            loss = MAIN_LOSS[config.MAIN_LOSS](projection, pair, y)
            if config.REG_LOSS:
                if config.REG_LOSS == 'Decorrelation':
                    representation = representation.split(config.BATCH_SIZE)[0].T
                    loss += config.DECORR_WEIGHT * REG_LOSS[config.REG_LOSS](representation, pair)
                else:
                    loss += REG_LOSS[config.REG_LOSS](projection, pair)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_loop.set_description(f'Loss: {loss.item():>8.4f}')
        
        # update learning rate
        lr_decay_steps = torch.sum(epoch > torch.Tensor(config.LR_DECAY_EPOCHS))
        if (lr_decay_steps > 0 and not(config.COSINE_DECAY or config.EXP_DECAY)):
            scheduler.gamma = config.LR_DECAY_RATE ** lr_decay_steps
        scheduler.step()
        
        features, labels = get_representations(net, dataloader_test)
        acc = lls(features, labels, dataset_test.n_classes)
        wb = wcss_bcss(features, labels, dataset_test.n_classes)
        pacmap_plot = get_pacmap(features, labels, epoch + 1,
            dataset_test.n_classes, dataset_test.labels)
        similarity_plot = get_neighbor_similarity(features, labels, epoch + 1)
        print(f"Method: {config.RUN_NAME.split('~')[0]}, Epoch: {epoch + 1}, "
              f"Read-Out Acc:{acc * 100:>6.2f}%, WCSS/BCSS:{wb:>8.4f}")

        # record results
        writer.add_scalar('accloss/loss', loss.item(), epoch + 1)
        writer.add_scalar('accloss/accuracy', acc, epoch + 1)
        writer.add_scalar('analytics/WCSS-BCSS', wb, epoch + 1)
        writer.add_scalar('analytics/learningrate', scheduler.get_last_lr()[0], epoch + 1)
        writer.add_figure('PacMap', pacmap_plot, epoch + 1)
        writer.add_figure('Object-Similarity', similarity_plot, epoch + 1)
    if config.SAVE_EMBEDDING:
        writer.add_embedding(features, tag='Embedding')


# ----------------
# main program
# ----------------

if __name__ == '__main__':
    for i in range(config.N_REPEAT):
        config.RUN_NAME = config.RUN_NAME.rsplit('~')[0] + f'~{i}'
        train()

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
