import argparse
import torch
import numpy as np
from utils import *
from train_model import *
from config import *

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) #--
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def main():
    args = load_arguments()
    config = eval(args.prototype)()

    print_config(config)
    set_random_seed(config['random_seed'])

    model = TRAIN_MODEL(config,args)

    if not args.eval:
        model.fit()
    model.test(ground_truth='PSCM', data='valid')
    model.test(ground_truth='PSCM',data='test')
    model.test(ground_truth='DBN', data='test')
    model.test(ground_truth='HUMAN',data='test')

if __name__ == '__main__':
    main()
