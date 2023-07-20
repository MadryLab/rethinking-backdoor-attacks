import os
import numpy as np
import torch as ch

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from argparse import ArgumentParser

Section('cfg', 'config file').params(
    dm_path=Param(str, 'path to datamodels matrix', required=True),
    save_path=Param(str, 'location to save results', required=True),
    num_trials=Param(int, 'number of trials for local search', default=100),
)

@param('cfg.dm_path')
def load_input(dm_path: ch.Tensor) -> ch.Tensor:
    if not os.path.exists(dm_path):
        raise FileNotFoundError

    input_matrix = ch.load(dm_path)
    input_matrix.fill_diagonal_(0.0)

    return input_matrix

@param('cfg.save_path')
@param('cfg.num_trials')
def main(save_path, num_trials):
    input_matrix = load_input().float().numpy()

    path = os.path.join(save_path, 'trials')
    os.makedirs(path, exist_ok=True)

    sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
    for size in sizes:
        result_path = os.path.join(path, f'result_{size}.npmap')
        result_mmap = np.lib.format.open_memmap(result_path, mode='w+', shape=(num_trials, input_matrix.shape[0]), dtype=np.uint8)
        result_mmap.flush()

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()