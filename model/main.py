import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import os

import models
import train
import configs
import evaluate
from load_data import load_data

torch.manual_seed(3407)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
    configs.get_singapore_time()
    logging.info('1/4: start to prepare the dataset.')
    
    # Load the data loaders
    train_iterator = load_data(configs.data_path + "/" + configs.train_file)
    valid_iterator = load_data(configs.data_path + "/" + configs.valid_file)
    test_iterator = load_data(configs.data_path + "/" + configs.test_file)

    configs.get_singapore_time()
    
    logging.info('2/4: start to construct our model.')
    model = models.LSTMwithLSTM11(
        desc_dim=100002, desc_embedding_dim=512, desc_hidden_dim=512, desc_n_layers=1,
        code_dim=100002, code_embedding_dim=512, code_hidden_dim=512, code_n_layers=1,
        dropout=0., hidden_dim=512, output_dim=1)
    
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer,'min',verbose=True,factor=0.1)
    criterion = nn.BCEWithLogitsLoss()

    if not configs.debug:
        model = torch.nn.DataParallel(model, device_ids=configs.gpus, output_device=configs.gpus[0])

    model.to(configs.device)
    
    configs.get_singapore_time()                                  
    logging.info('3/4: start to train.')
    
    train.train(
        model=model, train_iterator=train_iterator, valid_iterator=valid_iterator, 
        optimizer=optimizer, criterion=criterion, scheduler=scheduler, 
        num_epochs=35, eval_every=1000, save_every=50000, 
        file_path=configs.save_path, best_valid_loss=float('Inf')
    )
    
    configs.get_singapore_time()
    logging.info('4/4: start to evaluate')
    evaluate.evaluate(
        model=model, test_iterator=test_iterator, reload_from_checkpoint=True,
        load_path_checkpoint=os.path.join(configs.save_path,'model_f1.pt'),optimizer=optimizer
    )
    
    configs.get_singapore_time()
