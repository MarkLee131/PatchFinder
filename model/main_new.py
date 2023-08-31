'''
31/08/2023

In this script, we design the model by only using Codereviewer, i.e., we deprecate the use of LSTM for CVE description.

As you can see, them model is based on the pytorch-lightning framework.

'''


import configs
from load_data import CVEDataset
import logging
from torch.utils.data import DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
gpus = [0,1,2,3]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


####### Load the data loaders
configs.get_singapore_time()
logging.info('1/4: start to prepare the dataset.')

train_data = CVEDataset(configs.train_file)
valid_data = CVEDataset(configs.valid_file)
test_data = CVEDataset(configs.test_file)

train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=32, num_workers=10)
valid_dataloader = DataLoader(dataset=valid_data, batch_size=32, num_workers=10)
test_dataloader = DataLoader(test_data, batch_size=4, num_workers=10)

# batch = next(iter(train_dataloader))
# print(batch.keys())


"""## Fine-tune using PyTorch Lightning

As we will train the model using PyTorch Lightning, we first need to define a `LightningModule`, which is an `nn.Module` with some additional functionalities. We just need to define the `forward` pass, `training_step` (and optionally `validation_step` and `test_step`), and the corresponding dataloaders.
PyTorch Lightning will then automate the training for us, handling device placement (i.e. we don't need to type `.to(device)` anywhere), etc. It also comes with support for loggers (such as Tensorboard, Weights and Biases) and callbacks.

Of course, you could also train the model in other ways:
* using regular PyTorch
* using the HuggingFace Trainer (in this case, the Seq2SeqTrainer)
* using HuggingFace Accelerate
* etc.
"""

from transformers import AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch.nn as nn
import torch
# import torch.optim.AdamW as AdamW

class CVEClassifier(pl.LightningModule):
    def __init__(self, 
                 lstm_hidden_size,
                 num_classes,
                 lstm_layers=1,
                 dropout=0.1,
                 lstm_input_size=512,
                 lr=5e-5,
                 num_train_epochs=20,
                 warmup_steps=1000,
                 ):
        
        super().__init__()
        self.codeReviewer = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/codereviewer").encoder
        
        ### This is default setting.
        # # Set requires_grad=True for all parameters of codeReviewer to fine-tune it
        # for param in self.codeReviewer.parameters():
        #     param.requires_grad = True

        
        self.save_hyperparameters()
        
        # LSTM parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.lstm_input_size = lstm_input_size
        self.criterion = nn.BCEWithLogitsLoss()

        self.desc_embedding = nn.Embedding(32216, lstm_input_size)
        '''
        vocab size: 32216
        https://huggingface.co/microsoft/codereviewer/blob/main/config.json#L94
        '''

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # Fully connected layer for output, and take care of the bidirectional
        self.fc = nn.Linear(2 * self.lstm_hidden_size + 2 * self.codeReviewer.config.hidden_size, num_classes)

    def forward(self, input_ids_desc, attention_mask_desc, input_ids_msg,
                attention_mask_msg, input_ids_diff, attention_mask_diff):
        
        # Getting embeddings for all inputs
        desc_embed = self.desc_embedding(input_ids_desc)
        
        # Pass through LSTM and max-pooling
        lstm_output, _ = self.lstm(desc_embed)#, None)  # Passing None for hidden state and cell state will initialize them with 0s
        max_pooled, _ = torch.max(lstm_output, 1)  # Max pooling
        
        
        # Get [CLS] embeddings for msg and diff
        msg_cls_embed = self.codeReviewer(input_ids=input_ids_msg, attention_mask=attention_mask_msg).last_hidden_state[:, 0, :]
        diff_cls_embed = self.codeReviewer(input_ids=input_ids_diff, attention_mask=attention_mask_diff).last_hidden_state[:, 0, :]
        
        # Concatenate max-pooled LSTM output and [CLS] embeddings
        concatenated = torch.cat((max_pooled, msg_cls_embed, diff_cls_embed), dim=1)
        
        # Apply dropout
        dropped = self.dropout_layer(concatenated)
        
        # Pass through the fully connected layer
        output = self.fc(dropped)
        
        return output
    
    def common_step(self, batch, batch_idx):
        
        predict = self(
            batch['input_ids_desc'],
            batch['attention_mask_desc'],
            batch['input_ids_msg'],
            batch['attention_mask_msg'],
            batch['input_ids_diff'],
            batch['attention_mask_diff']
        )
        # ValueError: Target size (torch.Size([512])) must be the same as input size (torch.Size([512, 1]))
        predict = predict.squeeze(1)
        loss = self.criterion(predict, batch['label'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return valid_dataloader

    def test_dataloader(self):
        return test_dataloader

"""Let's start up Weights and Biases!"""

import wandb

wandb.login()
# 8f66cd17219a1912e8a14a65348e656c657f6c5e




"""Next, we initialize the model."""
configs.get_singapore_time()
###### Load the model ######
logging.info('2/4: start to construct our model.')


model = CVEClassifier(
        lstm_hidden_size=256,
        num_classes=1,   # binary classification
        lstm_layers=1,
        dropout=0.1,
        lstm_input_size=512,  # Assuming a 512-sized embedding
        # lr=5e-5, 
        # num_train_epochs=20, 
        # warmup_steps=1000,
    )


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import os


wandb_logger = WandbLogger(name='train', project='PatchSleuth')
# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=20,
    strict=False,
    verbose=False,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')

os.makedirs("/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints", exist_ok=True)


from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    monitor='validation_loss',
    dirpath='/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints',
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)


trainer = Trainer(
    accelerator='gpu',                    # "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto", "ddp"
    num_nodes=1,                          # Using 1 machine/node
    devices=len(gpus),                            # Using 4 GPUs
    default_root_dir="/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints",
    logger=wandb_logger,
    max_epochs=20,                        # Set the maximum number of epochs
    accumulate_grad_batches=1,           # Gradient accumulation steps
    max_steps=100000,                     # Set the maximum number of training steps
    log_every_n_steps=100,               # Log every 100 steps
    precision=32,                        # Using 32-bit precision for training; this is the default and can be omitted if desired
    gradient_clip_val=0.0,               # Assuming you don't want gradient clipping, but adjust if needed
    callbacks=[early_stop_callback, lr_monitor, checkpoint_callback],
)
torch.set_float32_matmul_precision("medium")

trainer.fit(model)

"""Once we're done training, we can also save the HuggingFace model as follows:"""
model_save_path = "/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints/final_model.pt"

# torch.save(model.state_dict(), model_save_path)

#### So we do not need to load the model Class when evaluate.
torch.save(model, model_save_path)
