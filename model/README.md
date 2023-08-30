# Deprecated


## Load data implementation

> We need to use the `torchtext` library to load data. The `torchtext` library has been reorganized in the latest version. The following code is based on the latest version of `torchtext` library. 
reference link: https://github.com/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb

### Steps

1. Train/validate/test split: generate train/validate/test data set if they are available
2. Tokenization: break a raw text string sentence into a list of words
3. Vocab: define a "contract" from tokens to indexes
4. Numericalize: convert a list of tokens to the corresponding indexes
5. Batch: generate batches of data samples and add padding if necessary

### Data format


### Notes



