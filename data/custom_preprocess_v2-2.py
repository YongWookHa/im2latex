from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    out_dir = Path(args.output)

    fn_list_jpg = list(Path(args.input).glob('*.jpg'))
    fn_list_png = list(Path(args.input).glob('*.png'))
    fn_list = fn_list_jpg + fn_list_png
    tqdm_bar = tqdm(fn_list, total=len(fn_list), desc='processing')
    for fn in tqdm_bar:
        img = cv2.imread(str(fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape

        board = np.ones((64, 544))*255

        # 64 - 6, 544 - 6  (3+3 for padding)
        if h > 58 or w > 538:
            ratio = min(58/h, 538/w)
            img = cv2.resize(img, dsize=(0, 0), fx=ratio, fy=ratio,
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape
        board[3:3+h,3:3+w] = img
        cv2.imwrite(str(out_dir/fn.name), board)
