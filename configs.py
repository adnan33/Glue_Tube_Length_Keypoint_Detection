import torch
import os

# constant paths
IMAGE_SHAPE = (512, 512)
TRAIN_PATH = './dataset/train'
TEST_PATH = './dataset/test'

# learning parameters
BATCH_SIZE = 8
LR = 0.0001
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# train/validation split
VAL_SPLIT = 0.09

root_dir = "model"
logdir = os.path.join(root_dir,"logs")
ckptdir = os.path.join(root_dir,"ckpt_dir")
os.makedirs(root_dir, exist_ok = True)
os.makedirs(logdir, exist_ok = True)
os.makedirs(ckptdir, exist_ok = True)