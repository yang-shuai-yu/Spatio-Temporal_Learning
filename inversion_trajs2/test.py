import torch
import os, sys, argparse, math, time
import numpy as np
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import random
import torch.nn.functional as F
import torch.optim as optim
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

def test(transformer, testLoader, device, args):
    print("Testing...")