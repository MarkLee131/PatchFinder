import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import os

import models
import train
import configs
import evaluate
from load_data import CVEDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

torch.manual_seed(3407)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    
    configs.get_singapore_time()
    logging.info('1/4: start to prepare the dataset.')
    
    # Load the data loaders
    train_data = CVEDataset(configs.train_file)
    valid_data = CVEDataset(configs.valid_file)
    test_data = CVEDataset(configs.test_file)
    
    train_data_loader = DataLoader(dataset=train_data,
                                      batch_size=configs.batch_size,
                                      shuffle=True,
                                      num_workers=48,
                                      drop_last=True)
    valid_data_loader = DataLoader(dataset=valid_data,
                                        batch_size=configs.batch_size,
                                        shuffle=True,
                                        num_workers=48,
                                        drop_last=True)
    test_data_loader = DataLoader(dataset=test_data,
                                        batch_size=configs.batch_size,
                                        shuffle=True,
                                        num_workers=48,
                                        drop_last=True)
    

    configs.get_singapore_time()
    
    ###### Load the model ######
    logging.info('2/4: start to construct our model.')

    # Modify model initialization to match the `CVEClassifier` signature.
    model = models.CVEClassifier(
        lstm_hidden_size=256,
        num_classes=1,   # binary classification
        lstm_layers=1,
        dropout=0.1,
        lstm_input_size=512  # Assuming a 512-sized embedding
    )
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer,'min',verbose=True,factor=0.1)
    criterion = nn.BCEWithLogitsLoss()

    if not configs.debug:
        model = torch.nn.DataParallel(model, device_ids=configs.gpus, output_device=configs.gpus[0])

    model.to(configs.device)
    
    configs.get_singapore_time()                                  
    logging.info('3/4: start to train.')
    
    # Assuming the train function from `train.py` expects DataLoader objects
    train.train(
        model=model, train_iterator=train_data_loader, valid_iterator=valid_data_loader, 
        optimizer=optimizer, criterion=criterion, scheduler=scheduler, 
        num_epochs=35, eval_every=1000, save_every=50000, 
        file_path=configs.save_path, best_valid_loss=float('Inf')
    )
    
    configs.get_singapore_time()
    logging.info('4/4: start to evaluate')

    # Assuming the evaluate function from `evaluate.py` expects DataLoader objects
    evaluate.evaluate(
        model=model, test_iterator=test_data_loader, reload_from_checkpoint=True,
        load_path_checkpoint=os.path.join(configs.save_path,'model_f1.pt'),optimizer=optimizer
    )
    
    configs.get_singapore_time()
