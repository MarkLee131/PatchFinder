import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'
gpus = [0,1]
data_path='/mnt/local/Predict5/data1'
train_file='train.csv'
valid_file='valid.csv'
test_file='test.csv'
batch_size=64
save_path='/mnt/local/Predict5/output7'
debug=False
device = torch.device("cuda" if torch.cuda.is_available() and not debug else 'cpu')
save_dataset=True #### whether the dataset is ready
alpha=0.05
gamma = 2