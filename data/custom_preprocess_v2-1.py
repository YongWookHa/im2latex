import argparse
from multiprocessing import Pool, Manager
from tqdm import tqdm
from pathlib import Path
from sympy import latex, preview

options = ["-T", "tight", "-z", "0", "--truecolor", "-D 600"]
args = None

def render_latex(mlist, tup):
    global args
    i, formula, font = tup
    formula = formula.strip('\n')
    if font:
        lat = '$$' + font + '{' + formula + '} $$'
    else:
        lat = '$$' + formula + '$$'
    try:
        preview(lat, viewer='file', filename='{}/{}.png'.format(args.output_dir, i),
                dvioptions=options)
        mlist.append((i, formula))
    except Exception:
        pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, required=True, dest='input_txt')
    parser.add_argument('--output_txt', type=str, required=True, dest='output_txt')
    parser.add_argument('--output_dir', type=str, required=True, dest='output_dir')
    args = parser.parse_args()

    Path('output').mkdir(exist_ok=True)

    with open(args.input_txt, 'r', encoding='utf8') as f:
        inp = f.readlines()
    inp = [(i, formula, '') for i, formula in enumerate(inp)]      \
        + [(i, formula, '\\mathrm') for i, formula in enumerate(inp, len(inp))] \
        + [(i, formula, '\\mathsf') for i, formula in enumerate(inp, len(inp)*2)] \
        + [(i, formula, '\\mathit') for i, formula in enumerate(inp, len(inp)*3)]
    print('len:', len(inp))

    p = Pool(36)
    m = Manager()
    mlist = m.list()
    p.starmap(render_latex, [(mlist, x) for x in inp])
    p.close()
    p.join()

    with open(args.output_txt, 'w', encoding='utf8') as f:
        for i, formula in mlist:
            f.write('{}.png\t{}\n'.format(i, formula))
