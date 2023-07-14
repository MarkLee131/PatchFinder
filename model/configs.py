import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
gpus = [0,1,2,3]
data_path='/mnt/local/Predict5/data1'
train_file='train.csv'
valid_file='valid.csv'
test_file='test.csv'
batch_size=128
save_path='/mnt/local/Predict5/output7'
debug=False
device = torch.device("cuda" if torch.cuda.is_available() and not debug else 'cpu')
save_dataset=False #### whether the dataset is ready
alpha=0.05
gamma = 2


import pytz
from datetime import datetime

### we define a function to get singapore time
def get_singapore_time():
    singaporeTz = pytz.timezone("Asia/Singapore") 
    timeInSingapore = datetime.now(singaporeTz)
    currentTimeInSinapore = timeInSingapore.strftime("%H:%M:%S")
    # print("Current time in Singapore is: ", currentTimeInSinapore)
    print(currentTimeInSinapore)