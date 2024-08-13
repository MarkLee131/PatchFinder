import torch
import configs
import numpy as np
import datetime
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def save_checkpoint(save_path, model, optimizer, valid_loss, epoch):

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

def train(model, train_loader, valid_loader, optimizer, criterion, num_epochs, eval_every, save_every, best_valid_loss, reload_from_checkpoint=False, load_path_checkpoint=None, load_path_metrics=None, save_path=None):
    
    device = configs.device
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []
    start_epoch = 0

    best_avg_recall_at_k = 0  # Initialize with 0
    best_avg_mrr = 0  # Initialize with 0

    if reload_from_checkpoint:
        valid_running_loss, start_epoch = load_checkpoint(load_path_checkpoint, model, optimizer)
        train_loss_list, valid_loss_list, global_steps_list = load_metrics(load_path_metrics)
        global_step = global_steps_list[-1]

    model.train()

    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch + 1}")
        
        for _, batch in enumerate(train_loader, 0):
            configs.get_singapore_time()
            # Extract batch data
            input_ids_desc = batch['input_ids_desc'].to(device) 
            attention_mask_desc = batch['attention_mask_desc'].to(device)
            input_ids_msg = batch['input_ids_msg'].to(device)
            attention_mask_msg = batch['attention_mask_msg'].to(device)
            input_ids_diff = batch['input_ids_diff'].to(device)
            attention_mask_diff = batch['attention_mask_diff'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass and calculate loss
            predict = model(input_ids_desc, attention_mask_desc, input_ids_msg, attention_mask_msg, input_ids_diff, attention_mask_diff)
            # ValueError: Target size (torch.Size([512])) must be the same as input size (torch.Size([512, 1]))
            predict = predict.squeeze(1)
            loss = criterion(predict, label)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % save_every == 0:
                model.eval()
                # Evaluate model
                avg_recall_at_k, avg_mrr = evaluate(model, valid_loader, criterion, configs.device)
                
                # Logging model performance
                logging.info(f'Epoch: {epoch+1}/{num_epochs}, Step: {global_step}, Train Loss: {running_loss / eval_every:.4f}, Valid Loss: {valid_running_loss / eval_every:.4f}')
                logging.info(f'Current avg_recall_at_k: {avg_recall_at_k:.4f}, Best avg_recall_at_k: {best_avg_recall_at_k:.4f}')
                logging.info(f'Current avg_mrr: {avg_mrr:.4f}, Best avg_mrr: {best_avg_mrr:.4f}')
                
                # Check if current metrics are better than the best ones observed so far
                if avg_recall_at_k > best_avg_recall_at_k and avg_mrr > best_avg_mrr:
                    best_avg_recall_at_k = avg_recall_at_k
                    best_avg_mrr = avg_mrr
                    
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    save_checkpoint(save_path + '/model_best_metric_' + timestamp + '.pt', model, optimizer, best_valid_loss, epoch)
                    save_metrics(save_path + '/metrics_best_metric_' + timestamp + '.pt', train_loss_list, valid_loss_list, global_steps_list)
                
                # Reset running losses for next evaluation
                running_loss = 0.0
                valid_running_loss = 0.0

                model.train()

    final_save_path = os.path.join(save_path, 'final.pt')
    save_checkpoint(final_save_path, model, optimizer, best_valid_loss, epoch)
    logging.info(f'Final model saved to ==> {final_save_path}')
    print('Finished Training!')
    
    
def evaluate(model, test_loader, k=10, reload_from_checkpoint=False, load_path_checkpoint=None, optimizer=None):
    model.eval()
    device = configs.device
    if reload_from_checkpoint:
        load_checkpoint(load_path_checkpoint, model, optimizer)
        logging.info(f'Model loaded from <== {load_path_checkpoint} for evaluation')


    results = {}

    with torch.no_grad():
        for _, batch in enumerate(test_loader, 0):
            logging.info(f"Processing batch {_ + 1}")
            
            input_ids_desc = batch['input_ids_desc'].to(device) 
            attention_mask_desc = batch['attention_mask_desc'].to(device)
            input_ids_msg = batch['input_ids_msg'].to(device)
            attention_mask_msg = batch['attention_mask_msg'].to(device)
            input_ids_diff = batch['input_ids_diff'].to(device)
            attention_mask_diff = batch['attention_mask_diff'].to(device)
            label = batch['label'].to(device)
            cve_ids = batch['cve'].to(device)
            predict = model(input_ids_desc, attention_mask_desc, input_ids_msg, attention_mask_msg, input_ids_diff, attention_mask_diff)
            y_scores = torch.sigmoid(predict).cpu().numpy()
            
            for i, cve_id in enumerate(cve_ids):
                if cve_id not in results:
                    results[cve_id] = {"scores": [], "labels": []}
                results[cve_id]["scores"].append(y_scores[i])
                results[cve_id]["labels"].append(label[i].item())

    total_recall_at_k = 0
    total_mrr = 0
    total_groups = 0

    if not results:
        logging.error("No results found during evaluation. Check the data loader.")
        return 0, 0
    
    for cve_id, data in results.items():
        labels = np.array(data["labels"])
        scores = np.array(data["scores"])

        # Sorting labels based on scores
        sorted_labels = labels[np.argsort(-scores)]

        # recall@k
        recall_at_k = sum(sorted_labels[:k]) / sum(labels)
        total_recall_at_k += recall_at_k

        # MRR
        rank = np.where(sorted_labels == 1)[0]
        if len(rank) > 0:
            total_mrr += (1. / (rank[0] + 1))
        
        total_groups += 1

    avg_recall_at_k = total_recall_at_k / total_groups
    avg_mrr = total_mrr / total_groups

    return avg_recall_at_k, avg_mrr
