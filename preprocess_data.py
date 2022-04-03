import numpy as np

def preprocess(average_size, convolved=False, random=False, random_count=3000, train_split=0.9, subsection='finetune'):

    assert not (convolved and random), "Either convolved or random should be True"
    assert subsection in ('finetune', 'reference'), "subsection should be either finetune or reference"

    X_fn = f'F:/Datasets/RAMAN_data/X_{subsection}.npy'
    y_fn = f'F:/Datasets/RAMAN_data/y_{subsection}.npy'
    X = np.load(X_fn)
    y = np.load(y_fn)

    X_train = []
    y_train = []

    X_val = []
    y_val = []

    split_by_value = []
    for i in np.unique(y):
        split_by_value.append((np.squeeze(np.argwhere(y == i)), i))

    for i, (idxs, value) in enumerate(split_by_value):
        val_set = X[idxs][int(len(idxs) * train_split):]
        X_val.append(val_set)
        y_val.append(np.ones(val_set.shape[0]) * value)
        train_set = X[idxs][:int(len(idxs) * train_split)]
        if random:
            rng = np.random.default_rng()
            for i in range(random_count):
                X_train.append(np.mean(rng.choice(train_set, size=average_size, replace=False), axis=0))
            y_train.append(np.ones(random_count) * value)
        else:
            if convolved:
                rangevals = train_set.shape[0] - average_size + 1
            else:
                rangevals = int(train_set.shape[0] / average_size)
            for j in range(rangevals):
                if convolved:
                    idx = list(range(j, j + average_size))
                else:
                    idx = list(range(j*average_size, (j + 1)*average_size))
                X_train.append(np.mean(train_set[idx], axis=0))
            y_train.append(np.ones(rangevals) * value)

    X_train = np.stack(X_train)
    X_val = np.concatenate(X_val)

    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)

    if convolved:
        filename = f'F:/Datasets/RAMAN_data/{subsection}_avg{average_size}_convolved.npz'
    elif random:
        filename = f'F:/Datasets/RAMAN_data/{subsection}_avg{average_size}_random.npz'
    else:
        filename = f'F:/Datasets/RAMAN_data/{subsection}_avg{average_size}.npz'
    np.savez(filename, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

if __name__ == '__main__':
    for average_size in [1, 2, 3, 4, 5]:
        for mode in ['normal', 'convolved', 'random']:
            preprocess(average_size, convolved=(True if mode == 'convolved' else False), random=(True if mode == 'random' else False), subsection='reference')
