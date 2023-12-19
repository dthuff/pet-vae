import argparse
import os
import re
from glob import glob

import imageio as iio
from pygifsicle import optimize


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def parse_cl_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("plot_dir")
    arg_parser.add_argument("save_dir")
    arg_parser.add_argument("--frame-rate")
    return arg_parser.parse_args()


def break_to_tiles(img_path, sz) -> list:
    img = iio.v3.imread(img_path)
    tiles = []
    for row in range(int(img.shape[0] / sz)):
        for col in range(0, int(img.shape[1] / sz), 2):
            tiles.append(img[sz * row:sz * (row + 1), sz * col:sz * (col + 2), :])
    return tiles


def save_gif(img_glob, save_path, loop=0):
    images = []
    for filename in img_glob:
        images.append(iio.v3.imread(filename))
    iio.mimwrite(save_path, images, loop=loop)


if __name__ == "__main__":
    cl_args = parse_cl_args()
    if not os.path.isdir(cl_args.save_dir):
        os.makedirs(cl_args.save_dir)

    # Save a gif of all tiles
    img_glob = glob(os.path.join(cl_args.plot_dir, "val_epoch*.png"))
    img_glob.sort(key=natural_keys)
    save_gif(img_glob, os.path.join(cl_args.save_dir, "all_tiles.gif"))
    # optimize(os.path.join(cl_args.save_dir, "all_tiles.gif"))

    # Make tiles
    for img in img_glob:
        epoch = os.path.basename(img).split("_")[2].split(".")[0]
        os.makedirs(os.path.join(cl_args.save_dir, f'epoch_{epoch}'))
        tiles = break_to_tiles(img, 150)
        for idx, tile in enumerate(tiles):
            iio.imsave(os.path.join(cl_args.save_dir, f'epoch_{epoch}', f'tile_{idx}.png'), tile)

    # Save a gif of each tile
    for i in range(32):
        img_glob = glob(os.path.join(cl_args.save_dir, "epoch*", f"tile_{i}.png"))
        img_glob.sort(key=natural_keys)
        save_gif(img_glob, os.path.join(cl_args.save_dir, f'tile_{i}.gif'))
        # optimize(os.path.join(cl_args.save_dir, f'tile_{i}.gif'))
