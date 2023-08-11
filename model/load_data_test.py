from torchtext.datasets import IMDB
train_iter, test_iter = IMDB(split=('train', 'test'))
print(next(train_iter))