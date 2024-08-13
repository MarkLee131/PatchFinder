import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
gpus = [0,1,2,3]

data_path='/mnt/local/Baselines_Bugs/PatchFinder/data'
save_dir='/mnt/local/Baselines_Bugs/PatchFinder/codellama/retrieval/'
os.makedirs(save_dir, exist_ok=True)


### 10/07
train_filename    = 'train_data.csv'
validate_filename = 'validate_data.csv'
test_filename     = 'test_data.csv'


train_file=os.path.join(data_path, train_filename)
valid_file=os.path.join(data_path, validate_filename)
test_file=os.path.join(data_path, test_filename)


debug=False
device = torch.device("cuda" if torch.cuda.is_available() and not debug else 'cpu')

import pytz
from datetime import datetime

### we define a function to get singapore time
def get_singapore_time():
    singaporeTz = pytz.timezone("Asia/Singapore") 
    timeInSingapore = datetime.now(singaporeTz)
    currentTimeInSinapore = timeInSingapore.strftime("%H:%M:%S")
    print(currentTimeInSinapore)