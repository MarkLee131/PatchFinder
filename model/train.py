import torch
import configs
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def save_checkpoint(save_path, model, optimizer, valid_loss,epoch):

    if save_path == None:
        return

    state_dict = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss'],state_dict['epoch']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def train(model, train_iterator, valid_iterator, optimizer, criterion, num_epochs, eval_every, save_every, best_valid_loss, reload_from_checkpoint=False, load_path_checkpoint=None, load_path_metrics=None):
    device = configs.device
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    start_epoch = 0
    best_f1 = 0
    alpha = torch.tensor([0.1, 0.9]).to(device)  # For focal loss

    if reload_from_checkpoint:
        valid_running_loss, start_epoch = load_checkpoint(load_path_checkpoint, model, optimizer)
        train_loss_list, valid_loss_list, global_steps_list = load_metrics(load_path_metrics)
        global_step = global_steps_list[-1]

    model.train()

    for epoch in range(start_epoch, num_epochs):
        for batch in train_iterator:
            code = batch.Code.to(device)  # The codes tensor
            msg = batch.Msg.to(device)    # The msgs tensor
            label = batch.Label.to(device)  # The labels tensor
            
            outputs = model(code, msg)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % eval_every == 0:
                model.eval()
                y_pred = []
                y_true = []
                y_scores = []

                with torch.no_grad():
                    for batch in valid_iterator:
                        cve = batch.cve.to(device)
                        diff_token = batch.diff_token.to(device)
                        msg_token = batch.msg_token.to(device)
                        label = batch.label.to(device)
                        
                        outputs = model(cve, diff_token, msg_token)
                        loss = criterion(outputs, label)
                        
                        valid_running_loss += loss.item()

                        y_scores.extend(torch.sigmoid(outputs).cpu())
                        y_pred.extend(torch.round(torch.sigmoid(outputs)).cpu())
                        y_true.extend(label.cpu())

                    average_loss = running_loss / eval_every
                    valid_average_loss = valid_running_loss / eval_every
                    train_loss_list.append(average_loss)
                    valid_loss_list.append(valid_average_loss)
                    global_steps_list.append(global_step)

                    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[1,0], zero_division=0)
                    logging.info('Epoch [{}/{}]:'.format(epoch+1, num_epochs))
                    logging.info('Train Loss: {:.4f}'.format(average_loss))
                    logging.info('Valid Loss: {:.4f}'.format(valid_average_loss))
                    logging.info('Precision: '+str(precision))
                    logging.info('Recall: '+str(recall))
                    logging.info('F1: '+str(f1))

                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    if best_valid_loss > valid_average_loss:
                        best_valid_loss = valid_average_loss
                        save_checkpoint(configs.save_path + '/model.pt', model, optimizer, best_valid_loss, epoch)
                        save_metrics(configs.save_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
                    if best_f1 < f1[1]:
                        best_f1 = f1[1]
                        save_checkpoint(configs.save_path + '/model_best_f1.pt', model, optimizer, best_valid_loss, epoch)
                        save_metrics(configs.save_path + '/metrics_best_f1.pt', train_loss_list, valid_loss_list, global_steps_list)
        
    save_metrics(configs.save_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')