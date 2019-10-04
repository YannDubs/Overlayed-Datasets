import numpy as np

import os


def overlay_save_datasets(
    bckgrnd_datasets, to_overlay_datasets, folder="data/", split_names=["train", "test"], **kwargs
):
    """Overlay corresponding train and test datasetsand save the output to file.
    
    Parameters
    ----------
    bckgrnd_datasets : tuple of tuple of arrays
        Background datasets on which to overlay the others. The exterior tuple corresponds to the 
        split (train, test, ...), the interior tuple is imgs/label: `((train_imgs, train_labels), ...)`, 
        image arrays should be of shape [n_bckgrnd, height_bckgrnd, width_bckgrnd, ...] and 
        dtype=uint8. Labels should be shape=[n_imgs, ...] and dtype=*.
        
    to_overlay_datasets : tuple of arrays
        Datasets to overlay. Same shape and form as the previous argument `bckgrnd_datasets`.

    folder : str, optional
        Folder to which to save the images.
        
    split_names : list of str, optional
        Names of all the splits, should be at least as long as len(bckgrnd_datasets).
        
    kwargs : 
        Additional arguments to `overlay_img`.
    """
    is_missing_names = len(split_names) < len(bckgrnd_datasets)
    if is_missing_names or len(bckgrnd_datasets) != len(to_overlay_datasets):
        err = "Sizes don't agree `len(split_names)={}, len(bckgrnd_datasets)={}, len(to_overlay_datasets)={}`."
        raise ValueError(
            err.format(len(split_names), len(bckgrnd_datasets), len(to_overlay_datasets))
        )

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, (bckgrnd, to_overlay, name) in enumerate(
        zip(bckgrnd_datasets, to_overlay_datasets, split_names)
    ):
        if to_overlay[0] is not None and bckgrnd[0] is not None:
            out, idcs = overlay_img(bckgrnd[0], to_overlay[0], **kwargs)

            np.save(os.path.join(folder, name + "_x.npy"), out, allow_pickle=False)
            np.save(os.path.join(folder, name + "_y.npy"), bckgrnd[1], allow_pickle=False)
            np.save(
                os.path.join(folder, name + "_y_distractor.npy"),
                to_overlay[1][idcs],
                allow_pickle=False,
            )


def overlay_img(bckgrnd, to_overlay, is_shift=False, seed=123):
    """Overlays an image with black background `to_overlay` on a `bckgrnd`
    
    Parameters
    ----------
    bckgrnd : np.array, shape=[n_bckgrnd, height_bckgrnd, width_bckgrnd, ...], dtype=uint8
        Background images. Each image will have one random image from  `to_overlay` overlayed on it.
    
    to_overlay : np.array, shape=[n_overlay, height_overlay, width_overlay, ...], dtype=uint8
        Images to overlay. Currently the following assumptions are made:
            - the overlaid images have to be at most as big as the background ones (i.e. 
              `height_bckgrnd <= height_bckgrnd` and `<= width_bckgrnd`).
            - The overlayed images are also used as mask. This is especially good for black 
              and white images : whiter pixels (~1) are the ones to be overlayed. In the case
              of colored image, this still hold but channel wise.

    is_shift : bool, optional
        Whether to randomly shift all overlayed images or to keep them on the bottom right.
        
    seed : int, optional
        Pseudo random seed.
        
    Return
    ------
    imgs : np.array, shape=[n_bckgrnd, height, width, 3], dtype=uint8
        Overlayed images.
        
    selected : np.array, shape=[n_bckgrnd], dtype=int64
        Indices of the slected overlayed images.
    """
    np.random.seed(seed)

    n_bckgrnd = bckgrnd.shape[0]
    n_overlay = to_overlay.shape[0]
    selected = np.random.choice(np.arange(n_overlay), size=n_bckgrnd)
    to_overlay = to_overlay[selected, ...]

    bckgrnd = ensure_color(bckgrnd).astype(np.float32)
    to_overlay = ensure_color(to_overlay).astype(np.float32)
    over_shape = to_overlay.shape[1:]
    bck_shape = bckgrnd.shape[1:]

    get_margin = lambda i: (bck_shape[i] - over_shape[i]) // 2
    get_max_shift = lambda i: get_margin(i) + over_shape[i] // 3
    get_shift = (
        lambda i: np.random.randint(-get_max_shift(i), get_max_shift(i))
        if is_shift
        else get_max_shift(i) // 2
    )

    resized_overlay = np.zeros((n_bckgrnd,) + bck_shape[:2] + over_shape[2:])
    resized_overlay[
        :, get_margin(0) : -get_margin(0) or None, get_margin(1) : -get_margin(1) or None
    ] = to_overlay

    for i in range(2):  # shift x and y
        resized_overlay = np.stack([np.roll(im, get_shift(i), axis=i) for im in resized_overlay])

    mask = resized_overlay / 255

    return (mask * resized_overlay + (1 - mask) * bckgrnd).astype(np.uint8), selected


def at_least_ndim(arr, ndim):
    """Ensures that a numpy array is at least `ndim`-dimensional."""
    padded_shape = arr.shape + (1,) * (ndim - len(arr.shape))
    return arr.reshape(padded_shape)


def ensure_color(imgs):
    """
    Ensures that a batch of colored (3 channels) or black and white (1 channels) numpy uint 8 images 
    is colored (3 channels).
    """
    imgs = at_least_ndim(imgs, 4)
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)
    return imgs

