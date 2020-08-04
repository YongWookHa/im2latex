# Im2LaTeX

Read Formula Image and translate to LaTeX Grammar, similar to [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) and [Harvard's paper and dataset](http://lstm.seas.harvard.edu/latex/).

I've changed the model structure based from [Show, Attend and Tell](https://arxiv.org/abs/1502.03044). 

## Overview

This repository is built on base-template of [Pytorch Template](https://github.com/YongWookHa/pytorch-template) which is bi-product of original [Pytorch Project Template](https://github.com/moemen95/Pytorch-Project-Template). Check the template repositories first before getting started.


The main difference from [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) is that I replaced row-encoder to positional encoding. And I set less 'max sequence length' with 40. With these changes, I could get `perplexity` of **1.0717** with reliable performance.

![](https://www.dropbox.com/s/r3iqjttuqi20jxt/im2latex_0.png?raw=1)  
im2latex Result = `\partial _ { \mu } ( F ^ { \mu \nu } - e j ^ { \mu } x ^ { \nu } ) : 0 .`

![](https://www.dropbox.com/s/fl1kteqtyd15y72/im2latex_1.png?raw=1)  
im2latex Result : `e x p \left( - \frac { \partial } { \partial \alpha _ { j } } \theta ^ { i k } \frac { \partial } { \partial \alpha _ { k } } \right)`

## Usage

### 1. Data Preprocess

Thanks to [untrix](https://github.com/untrix), we can get refined LaTeX dataset from [https://untrix.github.io/i2l/](https://untrix.github.io/i2l/).

He provides his data processing strategy, so you can follow his preprocessing steps. 
If you are in hurry, you can just download [Full Dataset](https://untrix.github.io/i2l/140k_download.html) as well.

Then you will have `Ascii-LaTeX Formula Text Datasets` around 140K formulas. Though you can get full formula images from untrix's dataset, I recommend to render the image yourself with LaTeX text dataset. 

You can use `sympy` library to render formula from LaTeX text. With `data/custom_preprocess_v2.py`, you can render two type of formula image with `Euler` font deciding variable.

### 2. Edit json configs file

If your data path is different, edit `configs/draft.json`.

```json
"debug": false,
"train_img_path" : "YOUR PATH",
"valid_img_path" : "YOUR PATH",
"train_formula_path" : "YOUR PATH",
"valid_formula_path" : "YOUR PATH"
```

### 3. Train  
Copy your `configs/draft.json` to `configs/train.json`.  
For training, you need to change the mode to `train`.
```json
# configs/train.json

"mode": "train",
```

In terminal, run `main.py` with your custom `train.json`.  

```shell  
python main.py configs/train.json  
```

### 4. Predict  
Copy your `configs/draft.json` to `configs/predict.json`.  

```json
"exp_name": "im2latex-draft",
"mode": "train",
"test_img_path" : "YOUR PATH",
"checkpoint_filename" : "YOUR PATH"
```

In terminal, run `main.py` with your custom `predict.json`.  

```shell
python main.py configs/predict.json
```

Enjoy the codes.