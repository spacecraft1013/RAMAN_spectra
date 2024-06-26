import itertools
import multiprocessing as mp
import os
from functools import partial
from time import time

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from datasets import spectral_dataloader
from preprocess_data import preprocess
from resnet import ResNet
from training import run_epoch


def train(params, subsection, use_pretrained):
    average_size, mode = params
    print_header = f"Average size: {average_size}, mode: {mode}"
    rank = mp.current_process()._identity[0]
    print(f"{print_header} - rank: {rank}")
    gpu = rank - 1
    t00 = time()
    writer = SummaryWriter(f"data/logs/avg{average_size}_{mode}")

    if average_size == 1:
        if mode != 'normal':
            return

    if mode == 'convolved':
        array_fn = f'F:/Datasets/RAMAN_data/{subsection}_avg{average_size}_convolved.npz'
    elif mode == 'random':
        array_fn = f'F:/Datasets/RAMAN_data/{subsection}_avg{average_size}_random.npz'
    else:
        array_fn = f'F:/Datasets/RAMAN_data/{subsection}_avg{average_size}.npz'

    if not os.path.exists(array_fn):
        preprocess(average_size, convolved=(True if mode == 'convolved' else False), random=(True if mode == 'random' else False), subsection=subsection)

    arrays = np.load(array_fn)

    X_train = arrays['X_train']
    y_train = arrays['y_train']

    X_train, y_train = sklearn.utils.shuffle(X_train, y_train)

    X_val = arrays['X_val']
    y_val = arrays['y_val']

    # CNN parameters
    layers = 6
    hidden_size = 100
    block_size = 2
    hidden_sizes = [hidden_size] * layers
    num_blocks = [block_size] * layers
    input_dim = 1000
    in_channels = 64
    n_classes = 30

    # Load trained weights for demo
    cnn = ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                in_channels=in_channels, n_classes=n_classes)
    cnn.to(gpu)
    if use_pretrained:
        cnn.load_state_dict(torch.load(
            './data/models/pretrained_model.ckpt', map_location=lambda storage, loc: storage))

    # Train CNN
    epochs = 100
    batch_size = 10
    t0 = time()
    # Set up Adam optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=1e-3, betas=(0.5, 0.999))
    # Set up dataloaders
    dl_tr = spectral_dataloader(X_train, y_train, idxs=None,
                                batch_size=batch_size, shuffle=True, num_workers=0)
    dl_val = spectral_dataloader(X_val, y_val, idxs=None,
                                batch_size=batch_size, shuffle=False, num_workers=0)
    # Fine-tune CNN for first fold
    best_val = 0
    no_improvement = 0
    max_no_improvement = 5
    for epoch in range(epochs):
        print(f'{print_header} Epoch {epoch+1}: {time()-t0:0.2f}s')
        # Train
        acc_tr, loss_tr = run_epoch(epoch, cnn, dl_tr,
                                    training=True, optimizer=optimizer, device=gpu)
        print(f'{print_header}  Epoch {epoch+1}  Train acc: {acc_tr:0.2f}')
        # Val
        acc_val, loss_val = run_epoch(epoch, cnn, dl_val,
                                    training=False, optimizer=optimizer, device=gpu)
        print(f'{print_header}  Epoch {epoch+1}  Val acc  : {acc_val:0.2f}')
        loss_data = {'Training': loss_tr, 'Validation': loss_val}
        accuracy_data = {'Training': acc_tr, 'Validation': acc_val}
        writer.add_scalars('Loss', loss_data, epoch+1)
        writer.add_scalars('Accuracy', accuracy_data, epoch+1)
        writer.flush()
        # Check performance for early stopping
        if acc_val > best_val or epoch == 0:
            best_val = acc_val
            no_improvement = 0
            best_weights = cnn.state_dict()
            best_acc = acc_val
            best_loss = loss_val
        else:
            no_improvement += 1
        if no_improvement >= max_no_improvement:
            print(f'{print_header} Finished after {epoch+1} epochs!')
            break

    print(
        f'\n Model Average: {average_size}, mode: {mode} completed in: {time()-t00:0.2f}s')

    if mode == 'convolved':
        outname = f'./data/models/model_{subsection}_average_{average_size}_convolved{"_noweights" if not use_pretrained else "_pretrained"}.ckpt'
    elif mode == 'random':
        outname = f'./data/models/model_{subsection}_average_{average_size}_random{"_noweights" if not use_pretrained else "_pretrained"}.ckpt'
    else:
        outname = f'./data/models/model_{subsection}_average_{average_size}{"_noweights" if not use_pretrained else "_pretrained"}.ckpt'

    torch.save({'weights': best_weights, 'train_acc': acc_tr, 'val_acc': best_acc},
            outname)

    return average_size, mode, acc_tr, best_acc, loss_tr, best_loss


if __name__ == '__main__':
    mp.set_start_method('spawn')

    size_search_space = [1, 2, 3, 4, 5]
    mode_search_space = ['normal', 'convolved', 'random']
    subsection = 'reference'
    use_pretrained = False
    model_params = []

    np.random.seed(0)

    for parameters in itertools.product(size_search_space, mode_search_space):
        model_params.append(parameters)

    with mp.Pool(min(torch.cuda.device_count(), len(model_params), mp.cpu_count())) as pool:
        out_data = pool.map(partial(train, subsection=subsection, use_pretrained=use_pretrained), model_params)

    average_size, mode, acc_tr, acc_val, loss_tr, loss_val = zip(*filter(None, out_data))

    results_df = pd.DataFrame({'Average Size': average_size,
                  'Mode': mode,
                  'Train Accuracy': acc_tr,
                  'Validation Accuracy': acc_val,
                  'Train Loss': loss_tr,
                  'Validation Loss': loss_val})

    results_df.to_csv(f'results{"_noweights" if not use_pretrained else "_pretrained"}_{subsection}_3000samples.csv', index=False)
