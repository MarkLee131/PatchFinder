import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
gpus = [0,1,2,3]

data_path='/mnt/local/Baselines_Bugs/PatchSleuth/TF-IDF/results/top_100_msg'
os.makedirs(data_path, exist_ok=True)


### 10/12
train_filename    = 'similarity_data_msg_train_top100.csv'
validate_filename = 'similarity_data_msg_validate_top100.csv'
test_filename     = 'similarity_data_msg_test_top100.csv'


train_file=os.path.join(data_path, train_filename)
valid_file=os.path.join(data_path, validate_filename)
test_file=os.path.join(data_path, test_filename)
batch_size=128


debug=False
device = torch.device("cuda" if torch.cuda.is_available() and not debug else 'cpu')

import pytz
from datetime import datetime

### we define a function to get singapore time
def get_singapore_time():
    singaporeTz = pytz.timezone("Asia/Singapore") 
    timeInSingapore = datetime.now(singaporeTz)
    currentTimeInSinapore = timeInSingapore.strftime("%H:%M:%S")
    # print("Current time in Singapore is: ", currentTimeInSinapore)
    print(currentTimeInSinapore)