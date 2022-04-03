import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from config import ATCC_GROUPINGS, ORDER, STRAINS, ab_order, antibiotics
from datasets import spectral_dataloader
from resnet import ResNet
from training import get_predictions

t00 = time()

arrays = np.load('F:/Datasets/RAMAN_data/reference_avg1.npz')
X = arrays["X_val"]
y = arrays["y_val"]


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

cnn = ResNet(hidden_sizes, num_blocks, input_dim,
             in_channels=in_channels, n_classes=n_classes)

if cuda:
    cnn.cuda()

for checkpoint_dir in os.listdir('./server_models'):
    print("Running " + checkpoint_dir)
    cnn.load_state_dict(torch.load(
        './server_models/' + checkpoint_dir, map_location=lambda storage, loc: storage)["weights"])


    t0 = time()
    dl = spectral_dataloader(X, y, batch_size=10, num_workers=0, shuffle=False)
    y_hat = get_predictions(cnn, dl, cuda)
    print('Predicted {} spectra: {:0.2f}s'.format(len(y_hat), time()-t0))

    acc = (y_hat == y).mean()
    print('Accuracy: {:0.1f}%'.format(100*acc))


    sns.set_context("talk", rc={"font": "Helvetica", "font.size": 12})
    label = [STRAINS[i] for i in ORDER]
    cm = confusion_matrix(y, y_hat, labels=ORDER)
    plt.figure(figsize=(15, 12))
    cm = 100 * cm / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm, annot=True, cmap='GnBu', fmt='0.0f',
                    xticklabels=label, yticklabels=label)
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.savefig('./figures/' + os.path.splitext(checkpoint_dir)[0] + '.png', dpi=1000, bbox_inches='tight', transparent=True)


    y_ab = np.asarray([ATCC_GROUPINGS[i] for i in y])
    y_ab_hat = np.asarray([ATCC_GROUPINGS[i] for i in y_hat])

    acc = (y_ab_hat == y_ab).mean()
    print('Accuracy: {:0.1f}%'.format(100*acc))

    sns.set_context("talk", rc={"font": "Helvetica", "font.size": 12})
    label = [antibiotics[i] for i in ab_order]
    cm = confusion_matrix(y_ab, y_ab_hat, labels=ab_order)
    plt.figure(figsize=(5, 4))
    cm = 100 * cm / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm, annot=True, cmap='GnBu', fmt='0.0f',
                    xticklabels=label, yticklabels=label)
    ax.xaxis.tick_top()
    plt.xticks(rotation=90)
    plt.savefig('./figures/' + os.path.splitext(checkpoint_dir)[0] + '_ab.png', dpi=1000, bbox_inches='tight', transparent=True)

    plt.clf()

print('\n This demo was completed in: {:0.2f}s'.format(time()-t00))
