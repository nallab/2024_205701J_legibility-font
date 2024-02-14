import random

import numpy as np
import torch
from dataloader import split_dataset_fonts
from eval import start_test
from options import get_parser
from train import start_train
from forward import start_forward

def torch_fix_seed(seed=314):
    print('fix seed')
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
  torch_fix_seed()
  parser = get_parser()
  opts = parser.parse_args()

  if (opts.phase == 'train'):
      start_train(opts)
  elif (opts.phase == 'test'):
      start_test(opts)
  elif (opts.phase == 'forward'):
      start_forward(opts)
  elif (opts.phase == 'split_datasets'):
    split_dataset_fonts(opts)

if __name__ == "__main__":
    main()
