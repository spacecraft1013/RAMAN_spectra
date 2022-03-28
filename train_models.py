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


def train(average_size, mode, subsection, use_pretrained):
    print(f"Average size: {average_size}, mode: {mode}")
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(0)
    cuda = torch.cuda.is_available()

    # Load trained weights for demo
    cnn = ResNet(hidden_sizes, num_blocks, input_dim=input_dim,
                in_channels=in_channels, n_classes=n_classes)
    if cuda:
        cnn.cuda()
    if use_pretrained:
        cnn.load_state_dict(torch.load(
            './data/models/pretrained_model.ckpt', map_location=lambda storage, loc: storage))

    # Train CNN
    epochs = 30
    batch_size = 10
    t0 = time()
    # Set up Adam optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=1e-3, betas=(0.5, 0.999))
    # Set up dataloaders
    dl_tr = spectral_dataloader(X_train, y_train, idxs=None,
                                batch_size=batch_size, shuffle=True)
    dl_val = spectral_dataloader(X_val, y_val, idxs=None,
                                batch_size=batch_size, shuffle=False)
    # Fine-tune CNN for first fold
    best_val = 0
    no_improvement = 0
    max_no_improvement = 5
    print('Starting fine-tuning!')
    for epoch in range(epochs):
        print(' Epoch {}: {:0.2f}s'.format(epoch+1, time()-t0))
        # Train
        acc_tr, loss_tr = run_epoch(epoch, cnn, dl_tr, cuda,
                                    training=True, optimizer=optimizer)
        print('  Train acc: {:0.2f}'.format(acc_tr))
        # Val
        acc_val, loss_val = run_epoch(epoch, cnn, dl_val, cuda,
                                    training=False, optimizer=optimizer)
        print('  Val acc  : {:0.2f}'.format(acc_val))
        loss_data = {'Training': loss_tr, 'Validation': loss_val}
        accuracy_data = {'Training': acc_tr, 'Validation': acc_val}
        writer.add_scalars('Loss', loss_data, epoch+1)
        writer.add_scalars('Accuracy', accuracy_data, epoch+1)
        writer.flush()
        # Check performance for early stopping
        if acc_val > best_val or epoch == 0:
            best_val = acc_val
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= max_no_improvement:
            print('Finished after {} epochs!'.format(epoch+1))
            break

    print(
        f'\n Model Average: {average_size}, mode: {mode} completed in: {time()-t00:0.2f}s')

    if mode == 'convolved':
        outname = f'./data/models/model_{subsection}_average_{average_size}_convolved{"_noweights" if not use_pretrained else "_pretrained"}.ckpt'
    elif mode == 'random':
        outname = f'./data/models/model_{subsection}_average_{average_size}_random{"_noweights" if not use_pretrained else "_pretrained"}.ckpt'
    else:
        outname = f'./data/models/model_{subsection}_average_{average_size}{"_noweights" if not use_pretrained else "_pretrained"}.ckpt'

    torch.save({'weights': cnn.state_dict(), 'train_acc': acc_tr, 'val_acc': acc_val},
            outname)

    return average_size, mode, acc_tr, acc_val, loss_tr, loss_val


if __name__ == '__main__':
    results_df = pd.DataFrame()
    size_search_space = [1, 2, 3, 4, 5]
    mode_search_space = ['normal', 'convolved', 'random']
    subsection = 'finetune'
    use_pretrained = False

    np.random.seed(0)

    for parameters in itertools.product(size_search_space, mode_search_space):
        average_size, mode = parameters

    with mp.Pool(min(torch.cuda.device_count(), len(parameters), mp.cpu_count())) as pool:
        out_data = pool.map(partial(train, subsection=subsection, use_pretrained=use_pretrained), parameters)

    average_size, mode, acc_tr, acc_val, loss_tr, loss_val = zip(*out_data)

    pd.DataFrame({'Average Size': average_size,
                  'Mode': mode,
                  'Train Accuracy': acc_tr,
                  'Validation Accuracy': acc_val,
                  'Train Loss': loss_tr,
                  'Validation Loss': loss_val})

    results_df.to_csv(f'results{"_noweights" if not use_pretrained else "_pretrained"}_{subsection}.csv', index=False)
