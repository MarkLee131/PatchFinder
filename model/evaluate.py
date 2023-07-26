import torch
import configs
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
from train import save_checkpoint, load_checkpoint, save_metrics, load_metrics
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def evaluate(model,test_iterator, reload_from_checkpoint=False, load_path_checkpoint=None,optimizer=None):
    device = configs.device
    y_pred = []
    y_true = []
    y_scores = []
    if reload_from_checkpoint:
        load_checkpoint(load_path_checkpoint, model, optimizer)
    model.eval()
    with torch.no_grad():
        for batch in test_iterator:
            code = batch.Code[0].to(device)  # The codes tensor
            code_len = batch.Code[1].to(device)  # The code lengths tensor
            msg = batch.Msg[0].to(device)  # The msgs tensor
            msg_len = batch.Msg[1].to(device)  # The msg lengths tensor
            label = batch.Label.to(device)  # The labels tensor
            cat=batch.Cat.to(device)
            rule=batch.Rule.to(device)
            priority=batch.Priority.to(device)
            rank=batch.Rank.to(device)
            fields_warning=batch.FieldWarning.to(device)
            field_code=batch.Field[0].to(device)
            field_code_len=batch.Field[1].to(device)
            predict = model(msg,code,msg_len,code_len,field_code,field_code_len,cat,rule,rank,priority,fields_warning).squeeze(1)
            y_scores.extend(torch.sigmoid(predict).cpu())
            y_pred.extend(torch.round(torch.sigmoid(predict)).cpu())
            y_true.extend(label.cpu())
        
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[1,0],zero_division=0)
    logging.info('precision: '+str(precision))
    logging.info('recall: '+str(recall))
    logging.info('F1: '+str(f1))
    logging.info('support: '+ str(support))

    # calculating top-10 recall
    top_10_recall = sum([1 for true, score in zip(y_true, y_scores) if true == 1 and score in sorted(y_scores, reverse=True)[:10]]) / sum(y_true)
    logging.info(f'Top-10 Recall: {top_10_recall}')

    # calculating MRR (Mean Reciprocal Rank)
    rank_list = sorted([(score, i) for i, score in enumerate(y_scores)], reverse=True)
    rank_index = [rank for score, rank in rank_list if y_true[rank] == 1]
    MRR = sum([1/(i+1) for i in rank_index]) / len(rank_index)
    logging.info(f'MRR: {MRR}')
