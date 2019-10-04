import matplotlib.pyplot as plt
import numpy as np
import skimage


def plot_grid(datas, labels=None, distractors=None, n_rows=3, n_cols=9, figsize=(11, 4), seed=123):
    """Plot a grid of randomly selected images.

    Parameters
    ----------
    datas : np.array, shape=[n_imgs, height, width, 3], dtype=uint8
        All possible images to plot.

    labels : np.array, shape=[n_imgs], dtype=*, optional
        Labels of the main image, used as title if given.

    distractors : np.array, shape=[n_imgs], dtype=*, optional
        Labels of the overlayed image, appended to titile if given.

    n_rows : int, optional
        Number of rows of the grid to plot.

    n_cols : int, optional  
        Number of columns of the grid to plot.

    figsize : tuple of int, optional
        Size of the resulting figure.

    seed : int, optional
        Pseudo random seed.
    """
    np.random.seed(seed)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    for i in range(n_rows):
        for j in range(n_cols):

            ax = axes[i, j]
            idx = np.random.randint(len(datas))
            im = datas[idx]

            ax.imshow(im)

            title = ""
            if labels is not None:
                title += "L:{} ".format(labels[idx].item())
            if distractors is not None:
                title += "D:{}".format(distractors[idx].item())
            if labels is not None or distractors is not None:
                ax.set_title(title)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def _batch_transform(batch, transform=lambda x: x, shape=(28, 28)):
    """Transforms a batch of images."""
    batch = list(batch)
    if shape is not None:
        batch = [skimage.transform.resize(im, shape, preserve_range=True) for im in batch]
    return np.stack([transform(im) for im in batch])


def _list_of_xy_to_transformed_np(pairs, **kwargs):
    """
    Converts a list of imgs and labels of np array to 2 np.arrays.
    Additionaly applies a transformation on X.
    """
    t1, t2 = zip(*pairs)
    return _batch_transform(np.array(t1), **kwargs), np.array(t2)

