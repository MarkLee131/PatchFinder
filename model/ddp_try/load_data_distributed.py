import os
import torch
import logging
import argparse
import random
import numpy as np
# from tqdm import tqdm
import multiprocessing
import time
# from itertools import cycle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
# from models import build_or_load_gen_model
from model.ddp_try.configs_distributed import add_args, set_seed, set_dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# from utils import CommentClsDataset, SimpleClsDataset
# from sklearn.metrics import f1_score, accuracy_score
################
import configs
from load_data import CVEDataset
import models
# import torch.optim as optim
import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)





# os.environ['WORLD_SIZE'] = '4'  # for example, if you have 4 GPUs
# os.environ['RANK'] = '0'  # this needs to be different for each process



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# def get_loaders(data_files, args, tokenizer, pool, eval=False):
def get_loaders(data_files, args, eval=False):   
    def fn(features):
        return features
    global_rank = args.global_rank
    for data_file in data_files:
        # if args.raw_input:
        #     dataset = SimpleClsDataset(tokenizer, pool, args, data_file)
        # else:
        #     dataset = CommentClsDataset(tokenizer, pool, args, data_file)
        dataset = CVEDataset(data_file)
        
        data_len = len(dataset)
        if global_rank == 0:
            logger.info(f"Data length: {data_len}.")
        if eval:
            sampler = SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size if not eval else args.eval_batch_size, \
                                # num_workers=args.cpu_count, \
                                    num_workers=10, \
                                    collate_fn=fn)
        yield dataset, sampler, dataloader



def eval_epoch_acc_mrr(args, eval_dataloader, model):
    # Start evaluating model
    logger.info("  " + "***** Running acc evaluation and calculate mrr *****")
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    local_rank = 0
    results = {}
    k=10
    
    with torch.no_grad():
        for step, examples in enumerate(eval_dataloader, 1):
            # source_ids = torch.tensor(
            #     [ex.source_ids for ex in examples], dtype=torch.long
            # ).to(local_rank)
            # source_mask = source_ids.ne(tokenizer.pad_id)
            input_ids_desc = examples['input_ids_desc'].to(local_rank)
            attention_mask_desc = examples['attention_mask_desc'].to(local_rank)
            input_ids_msg = examples['input_ids_msg'].to(local_rank)
            attention_mask_msg = examples['attention_mask_msg'].to(local_rank)
            input_ids_diff = examples['input_ids_diff'].to(local_rank)
            attention_mask_diff = examples['attention_mask_diff'].to(local_rank)
            label = examples['label'].to(local_rank)
            cve_ids = examples['cve'].to(local_rank)
            
            predict = model(
                input_ids_desc, 
                attention_mask_desc,
                input_ids_msg,
                attention_mask_msg,
                input_ids_diff,
                attention_mask_diff
            )
            
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



# def save_model(model, optimizer, scheduler, output_dir, config):
def save_model(model, optimizer, scheduler, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    # config.save_pretrained(output_dir)
    output_model_file = os.path.join(output_dir, "final_model.pt")
    torch.save(model_to_save.state_dict(), output_model_file)
    output_optimizer_file = os.path.join(output_dir, "optimizer.pt")
    torch.save(
        optimizer.state_dict(),
        output_optimizer_file,
        _use_new_zipfile_serialization=False,
    )
    output_scheduler_file = os.path.join(output_dir, "scheduler.pt")
    torch.save(
        scheduler.state_dict(),
        output_scheduler_file,
        _use_new_zipfile_serialization=False,
    )


def main(args):
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % args.gpu_per_node
    args.global_rank = local_rank + args.node_index * args.gpu_per_node
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    logger.warning("Process rank: %s, global rank: %s, world size: %s, bs: %s",
                   args.local_rank, args.global_rank, \
                   torch.distributed.get_world_size(), \
                   args.train_batch_size)
    torch.cuda.set_device(local_rank)
    
    configs.get_singapore_time()
    # t0 = time.time()
    # set_dist(args)
    set_seed(args)
    
    # config, model, tokenizer = build_or_load_gen_model(args)
    
    #########################################################################
    # Modify model initialization to match the `CVEClassifier` signature.
    model = models.CVEClassifier(
        lstm_hidden_size=256,
        num_classes=1,   # binary classification
        lstm_layers=1,
        dropout=0.1,
        lstm_input_size=512  # Assuming a 512-sized embedding
    )
    
    # optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # scheduler = ReduceLROnPlateau(optimizer,'min',verbose=True,factor=0.1)
    criterion = nn.BCEWithLogitsLoss()

    # if not configs.debug:
    #     model.cuda()
    #     model = torch.nn.DataParallel(model, device_ids=configs.gpus, output_device=configs.gpus[0])

    # model.to(configs.device)
    
    #######################################################
    
    
    
    
    # load last model
    # if os.path.exists("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir)):
    #     model.load_state_dict(
    #         torch.load("{}/checkpoints-last/pytorch_model.bin".format(args.output_dir))
    #     )
    if os.path.exists(os.path.join(args.output_dir, "final_model.pt")):
        model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "final_model.pt"))
    )    
    
    
    
    model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # pool = multiprocessing.Pool(args.cpu_count)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    # if os.path.exists("{}/checkpoints-last/optimizer.pt".format(args.output_dir)):
    #     optimizer.load_state_dict(
    #         torch.load(
    #             "{}/checkpoints-last/optimizer.pt".format(args.output_dir),
    #             map_location="cpu",
    #         )
    #     )
    #     scheduler.load_state_dict(
    #         torch.load(
    #             "{}/checkpoints-last/scheduler.pt".format(args.output_dir),
    #             map_location="cpu",
    #         )
    #     )
    if os.path.exists(os.path.join(args.output_dir, "final_optimizer.pt")):
        optimizer.load_state_dict(
        torch.load(os.path.join(args.output_dir, "final_optimizer.pt"), map_location="cpu")
    )
        scheduler.load_state_dict(
        torch.load(os.path.join(args.output_dir, "final_scheduler.pt"), map_location="cpu")
    )
    
    global_step = 0
    save_steps = args.save_steps
    train_file = args.train_filename
    valid_file = args.dev_filename
    if os.path.isdir(train_file):
        train_files = [file for file in os.listdir(train_file) if file.startswith("cls-train-chunk") and file.endswith(".jsonl")]
    else:
        train_files = [train_file]
    logger.warning("Train files: %s", train_files)
    random.seed(args.seed)
    random.shuffle(train_files) ### no effect when train_file is a file
    ### fix by kaixuan
    if os.path.isdir(train_file):
        train_files = [os.path.join(train_file, file) for file in train_files]
    valid_files = [valid_file]
    for epoch in range(1, args.train_epochs + 1):
        configs.get_singapore_time()
        # set seed for reproducible data split
        save_seed = args.seed
        args.seed += epoch
        set_seed(args)
        args.seed = save_seed
        model.train()
        # nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
        nb_tr_steps, tr_loss = 0, 0
        for _, _, train_dataloader in get_loaders(train_files, args):        # WARNING: this is an iterator, to save memory
            for step, examples in enumerate(train_dataloader, 1):
                if step == 1:
                    # ex = examples[0]
                    logger.info(f"batch size: {len(examples)}")
                    # logger.info(f"example source: {tokenizer.convert_ids_to_tokens(ex.source_ids)}")
                    
                # source_ids = torch.tensor(
                #     [ex['source_ids'] for ex in examples], dtype=torch.long
                # ).to(local_rank)
                # ys = torch.tensor(
                #     [ex.y for ex in examples], dtype=torch.long
                # ).to(local_rank)
                # source_mask = source_ids.ne(tokenizer.pad_id)
                configs.get_singapore_time()
                # # Extract batch data
                input_ids_desc = examples['input_ids_desc'].to(local_rank)
                attention_mask_desc = examples['attention_mask_desc'].to(local_rank)
                input_ids_msg = examples['input_ids_msg'].to(local_rank)
                attention_mask_msg = examples['attention_mask_msg'].to(local_rank)
                input_ids_diff = examples['input_ids_diff'].to(local_rank)
                attention_mask_diff = examples['attention_mask_diff'].to(local_rank)
                label = examples['label'].to(local_rank)
                
                # loss = model(
                #     cls=True,
                #     input_ids=source_ids,
                #     labels=ys,
                #     attention_mask=source_mask
                # )
                
                # Forward pass and calculate loss
                predict = model(
                    input_ids_desc, 
                    attention_mask_desc,
                    input_ids_msg,
                    attention_mask_msg,
                    input_ids_diff,
                    attention_mask_diff
                    )
                # ValueError: Target size (torch.Size([512])) must be the same as input size (torch.Size([512, 1]))
                predict = predict.squeeze(1)
                loss = criterion(predict, label)

                if args.gpu_per_node > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                # nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    if args.global_rank == 0 and global_step % args.log_steps == 0:
                        train_loss = round(
                            tr_loss * args.gradient_accumulation_steps / nb_tr_steps,
                            4,
                        )
                        logger.info(
                            "step {}/{}: Train loss {}".format(
                                global_step,
                                args.train_steps,
                                round(train_loss, 3),
                            )
                        )
                if args.global_rank == 0 and global_step == args.train_steps:
                    # end training
                    _, _, valid_dataloader = next(get_loaders(valid_files, args, eval=True))
                    acc, mrr = eval_epoch_acc_mrr(args, valid_dataloader, model)
                    output_dir = os.path.join(args.output_dir, "checkpoints-last" + "-" + str(acc)[:5] + "-" + str(mrr)[:5])
                    os.makedirs(output_dir, exist_ok=True)
                    save_model(model, optimizer, scheduler, output_dir)
                    logger.info(f"Reach max steps {args.train_steps}.")
                    time.sleep(5)
                    return
                if args.global_rank == 0 and \
                        global_step % save_steps == 0 and \
                        nb_tr_steps % args.gradient_accumulation_steps == 0:
                    _, _, valid_dataloader = next(get_loaders(valid_files, args, eval=True))
                    acc, mrr = eval_epoch_acc_mrr(args, valid_dataloader, model)
                    output_dir = os.path.join(args.output_dir, "checkpoints-" + str(global_step) \
                        + "-" + str(acc)[:5] + "-" + str(mrr)[:5])
                    
                    os.makedirs(output_dir, exist_ok=True)
                    save_model(model, optimizer, scheduler, output_dir)
                    logger.info(
                        "Save the {}-step model and optimizer into {}".format(
                            global_step, output_dir
                        )
                    )
                    time.sleep(5)



def run():
    try:
        parser = argparse.ArgumentParser()
        args = add_args(parser)
        args.cpu_count = multiprocessing.cpu_count()
        # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        
        args.train_epochs = 20
        args.train_batch_size = configs.batch_size
        args.eval_batch_size = configs.batch_size
        args.gradient_accumulation_steps = 1
        args.learning_rate = 5e-5
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0
        args.weight_decay = 0.01
        args.train_steps = 100000
        args.save_steps = 1000
        args.log_steps = 100
        args.output_dir = configs.save_path
        args.train_filename = configs.train_file
        args.dev_filename = configs.valid_file
        args.max_seq_length = 512
        args.raw_input = False
        args.gpu_per_node = 1
        args.node_index = 0
        args.seed = 3407
        logger.info(args)
        main(args)
        logger.info("Training finished.")
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # remove long tokenization warning. ref: https://github.com/huggingface/transformers/issues/991
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    args.train_epochs = 20
    args.train_batch_size = configs.batch_size
    args.eval_batch_size = configs.batch_size
    args.gradient_accumulation_steps = 1
    args.learning_rate = 5e-5
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0
    args.weight_decay = 0.01
    args.train_steps = 100000
    args.save_steps = 1000
    args.log_steps = 100
    args.output_dir = configs.save_path
    args.train_filename = configs.train_file
    args.dev_filename = configs.valid_file
    args.max_seq_length = 512
    args.raw_input = False
    args.gpu_per_node = 1
    args.node_index = 0
    args.seed = 3407
    logger.info(args)
    main(args)
    logger.info("Training finished.")
    
    
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())