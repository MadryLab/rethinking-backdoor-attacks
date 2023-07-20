import os
import numpy as np
import torch as ch

import numba as nb
from numba.typed import Dict

from tqdm import tqdm
from functools import partial

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from argparse import ArgumentParser

Section('cfg', 'config file').params(
    dm_path=Param(str, 'path to datamodels matrix', required=True),
    save_path=Param(str, 'location to save results', required=True),
    num_trials=Param(int, 'number of trials for local search', default=100),
)

NB_PARALLEL = True
PARALLEL_ITER = nb.prange if NB_PARALLEL else range

@param('cfg.dm_path')
def load_input(dm_path: ch.Tensor) -> ch.Tensor:
    if not os.path.exists(dm_path):
        raise FileNotFoundError

    input_matrix = ch.load(dm_path)
    input_matrix.fill_diagonal_(0.0)

    return input_matrix

def random_start(A, size=100):
    return np.random.choice(A.shape[0], size=size, replace=False)

@nb.njit(parallel=NB_PARALLEL)
def compute_score(A, selection):
    result = 0
    for i in PARALLEL_ITER(len(selection)):
        x = selection[i]
        temp = 0
        for j in range(len(selection)):
            y = selection[j]
            temp += A[x, y]

        result += temp

    for i in PARALLEL_ITER(len(selection)):
        result -= A[i,i]

    result = result * 2
    return result

@nb.njit(parallel=NB_PARALLEL)
def hsum(A, selection, result):
    for i in PARALLEL_ITER(result.shape[0]):
        x = i
        for j in range(len(selection)):
            y = selection[j]
            result[i] += A[x, y]
    return result

@nb.njit(parallel=NB_PARALLEL)
def vsum(A, selection, result):
    for j in PARALLEL_ITER(result.shape[0]):
        y = j
        for i in range(len(selection)):
            x = selection[i]
            result[j] += A[x, y]
    return result

@nb.njit(parallel=NB_PARALLEL)
def hsum_correct(A, result, old_index, new_index):
    for i in PARALLEL_ITER(result.shape[0]):
        x = i
        y = old_index
        result[i] -= A[x,y]

    for i in PARALLEL_ITER(result.shape[0]):
        x = i
        y = new_index
        result[i] += A[x,y]

    return result

@nb.njit(parallel=NB_PARALLEL)
def vsum_correct(A, result, old_index, new_index):
    for j in PARALLEL_ITER(result.shape[0]):
        x = old_index
        y = j
        result[j] -= A[x,y]

    for j in PARALLEL_ITER(result.shape[0]):
        x = new_index
        y = j
        result[j] += A[x,y]
    return result

@nb.njit(parallel=NB_PARALLEL)
def total_sum_correct(A, result, old_index, new_index):
    vsum_correct(A, result, old_index, new_index)
    hsum_correct(A, result, old_index, new_index)

    for i in PARALLEL_ITER(result.shape[0]):
        result[i] += A[old_index, old_index]
        result[i] -= A[new_index, new_index]

    return result

@nb.njit(parallel=NB_PARALLEL)
def total_sum(A, selection):
    result = np.zeros(A.shape[0])
    vsum(A, selection, result)
    hsum(A, selection, result)

    for i in PARALLEL_ITER(result.shape[0]):
        x = i
        result[i] -= A[x,x]

    return result

@nb.njit(parallel=False)
def compute_diffs(A, selection, veto, scores):
    diff_taken = np.zeros(selection.shape[0])
    taken_ixes = np.zeros(selection.shape[0], dtype=np.int32)
    diff_not_taken = np.zeros(A.shape[0] - selection.shape[0])
    not_taken_ixes = np.zeros(diff_not_taken.shape[0], dtype=np.int32)
    s = set(selection)
    a = 0
    b = 0
    for i in range(A.shape[0]):
        if i in s:
            diff_taken[a] = scores[i]
            taken_ixes[a] = i
            a += 1
        else:
            diff_not_taken[b] = scores[i]
            not_taken_ixes[b] = i
            b += 1

    t = min(len(diff_taken), len(diff_not_taken))
    improvement = 0

    for ix in range(len(taken_ixes)):
        i = taken_ixes[ix]
        if i in veto:
            diff_taken[ix] = np.inf

    for ix in range(len(not_taken_ixes)):
        i = not_taken_ixes[ix]
        if i in veto:
            diff_not_taken[ix] = -np.inf

    best_removal_ix = np.argmin(diff_taken)
    best_addition_ix = np.argmax(diff_not_taken)
    removed = taken_ixes[best_removal_ix]
    added = not_taken_ixes[best_addition_ix]
    delta = diff_not_taken[best_addition_ix] - diff_taken[best_removal_ix]
    rix = np.where(selection == removed)[0]
    selection[rix] = added

    total_sum_correct(A, scores, removed, added)

    return removed, added, delta

def optimize(A, size=100, start=None):
    current = random_start(A, size)
    if start is not None:
        for i, v in enumerate(start):
            current[i] = v
    last = 0
    for epoch in range(100):
        selection = current.copy()
        best_score = 0
        best_state = selection.copy()
        veto = Dict()
        veto[5] = 0
        del veto[5]
        if start is not None:
            for v in start:
                veto[v] = 0
        current_diff = 0
        for i in (range(100)):
            if i == 0:
                scores = total_sum(A, selection)

            removed, added, delta = compute_diffs(A, selection, veto, scores)
            veto[added] = 1
            veto[removed] = 1
            current_diff += delta
            if current_diff > best_score:
                best_score = current_diff
                best_state = selection.copy()
        current = best_state.copy()

        score = compute_score(A, current)
        if score <= last:
            break
        last = score

    return current

def work(index, matrix, size, result_matrix):
    result = optimize(matrix, size=size, start=None)

    result_mask = np.zeros(matrix.shape[0], dtype=np.uint8)
    result_mask[result] = 1

    result_matrix[index, :] = result_mask

@param('cfg.save_path')
@param('cfg.num_trials')
def main(save_path, num_trials):
    input_matrix = load_input().float().numpy()
    path = os.path.join(save_path, 'trials')

    sizes = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]

    size_iter = tqdm(sizes, desc='Mask Size')
    for size in size_iter:

        size_iter.set_description_str(f'Size: {size}')

        result_path = os.path.join(path, f'result_{size}.npmap')
        result_mmap = np.lib.format.open_memmap(result_path, mode='r+')

        A_size = input_matrix.copy()
        np.fill_diagonal(A_size, -input_matrix.sum(1) * (size / input_matrix.shape[0]))

        result_size = np.zeros(shape=(num_trials, input_matrix.shape[0]), dtype=np.uint8)

        results_work = partial(
            work,
            matrix=A_size,
            size=size,
            result_matrix=result_size,
        )

        for i in tqdm(range(num_trials)):
            results_work(i)

        result_mmap[:] = result_size
        result_mmap.flush()
        del A_size

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    main()