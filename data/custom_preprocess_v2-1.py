import argparse
from pathlib import Path
from sympy import latex, preview


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, required=True, dest='input_txt')
    args = parser.parse_args()

    cnt = 0
    Path('output').mkdir(exist_ok=True)
    options = ["-T", "tight", "-z", "0", "--truecolor", "-D 600"]
    with open(args.input_txt, 'r', encoding='utf8') as f:
        for line in f:
            lat = '$$'+ line.strip() + '$$'
            preview(lat, viewer='file', filename='output/{}.png'.format(cnt), 
                    euler=True, dvioptions=options)
            cnt+=1

    with open(args.input_txt, 'r', encoding='utf8') as f:
        for line in f:
            lat = '$$'+ line.strip() + '$$'
            preview(lat, viewer='file', filename='output/{}.png'.format(cnt), 
                    euler=False, dvioptions=options)
            cnt+=1