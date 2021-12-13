#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import argparse
import datetime

# parse args that correspond to configurations to be experimented on
parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    default='',
                    type=str)
parser.add_argument('--n_fix',
                    default=2,
                    type=int)
parser.add_argument('--n_negative',
                    default=None,
                    type=int)
parser.add_argument('--dataset',
                    default='TDW',
                    choices=['TDW', 'Miyashita', 'COIL100'], type=str)
parser.add_argument('--main_loss',
                    default='SimCLR',
                    choices=['SimCLR', 'BYOL', 'supervised'], type=str)
parser.add_argument('--reg_loss',
                    default=None,
                    choices=[None, 'RELIC', 'Decorrelation', 'Max_entropy'], type=str)
parser.add_argument('--lrate',
                    default=1e-3,
                    type=float)
parser.add_argument('--cosine_decay',
                    dest='cosine_decay',
                    action='store_true')
parser.set_defaults(cosine_decay=False)
parser.add_argument('--exp_decay',
                    dest='exp_decay',
                    action='store_true')
parser.set_defaults(exp_decay=False)
parser.add_argument('--lrate_decay',
                    default=0.3,
                    type=float)
parser.add_argument('--decorr_weight',
                    default=0.4,
                    type=float)
parser.add_argument('--temperature',
                    default=1.,
                    type=float)
parser.add_argument('--similarity',
                    default='cosine',
                    choices=['cosine', 'RBF'], type=str)
parser.add_argument('--view_sampling',
                    default='randomwalk',
                    choices=['randomwalk', 'uniform', 'window'], type=str)
parser.add_argument('--shuffle_objects',
                    dest='shuffle_object_order',
                    action='store_true')
parser.add_argument('--no-shuffle_objects',
                    dest='shuffle_object_order',
                    action='store_false')
parser.set_defaults(shuffle_object_order=True)

parser.add_argument('--save_model',
                    default=False,
                    type=bool)
parser.add_argument('--save_embedding',
                    default=False,
                    type=bool)

args = parser.parse_args()

N_fix = args.n_fix
N_negative = args.n_negative
DATASET = args.dataset
MAIN_LOSS = args.main_loss
REG_LOSS = args.reg_loss
SIMILARITY = args.similarity
SHUFFLE_OBJECT_ORDER = args.shuffle_object_order
VIEW_SAMPLING = args.view_sampling
SAVE_MODEL = args.save_model
SAVE_EMBEDDING = args.save_embedding
LRATE = args.lrate
LR_DECAY_RATE = args.lrate_decay
DECORR_WEIGHT = args.decorr_weight
COSINE_DECAY = args.cosine_decay
EXP_DECAY = args.exp_decay
TEMPERATURE = args.temperature
DATASET = args.dataset
RUN_NAME = f'{datetime.datetime.now().strftime("%d-%m-%y_%H:%M")}_{args.name}_{DATASET}_aug_time_{MAIN_LOSS}_reg_{REG_LOSS}_nfix_{N_fix}'

# configurations that are not tuned
FEATURE_DIM = 128
HIDDEN_DIM = 256
LOG_DIR = 'save'
N_EPOCHS = 100
N_REPEAT = 5
DEVICE = 'cuda' #switch to 'cpu' for local testing
PRIOR = 'gaussian'
TAU = 0.996
BATCH_SIZE = 256
CROP_SIZE = 32

LR_DECAY_EPOCHS = [0] #[700, 800, 900]

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
