import models
import torch
import prepare_data_feature1
import train
import configs
import torch.optim as optim
import torch.nn as nn
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
torch.manual_seed(3407)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'
if __name__ == '__main__':
    logging.info('start contruct data iterator')
    train_iterator, vaild_iterator, test_iterator = prepare_data_feature1.prepare_data()
    
    logging.info('start contruct model')
    
    model = models.LSTMwithLSTM11(desc_dim=100002, desc_embedding_dim=512, desc_hidden_dim=512, desc_n_layers=1, 
                 code_dim=100002, code_embedding_dim=512, code_hidden_dim=512, code_n_layers=1,
                 dropout=0., hidden_dim=512, output_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = ReduceLROnPlateau(optimizer,'min',verbose=True,factor=0.1)
    criterion = nn.BCEWithLogitsLoss()
    if not configs.debug:
        model.cuda()
        model = torch.nn.DataParallel(model.cuda(), device_ids=configs.gpus, output_device=configs.gpus[0])
    model.to(configs.device)                                       
    logging.info('start train')
    train.train(model=model, train_iterator=train_iterator, vaild_iterator=vaild_iterator, optimizer=optimizer,criterion=criterion,scheduler=scheduler, num_epochs=35, eval_every=1000,save_every=50000, 
                file_path=configs.save_path, best_valid_loss=float('Inf'))
    logging.info('start evaluate')
    train.evaluate(model=model, test_iterator=test_iterator,reload_from_checkpoint=True,
                load_path_checkpoint=os.path.join(configs.save_path,'model_f1.pt'),optimizer=optimizer)