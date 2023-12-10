'''
31/08/2023
In this script, we design the model by only using Codereviewer. As you can see, 
the model is based on the pytorch-lightning framework.

04/10/2023
we reuse this script to train the model with the new top100 dataset (colbert).

07/10/2023
we reuse this script to train the model with the new top100 dataset (colbert) with 20 epoch for spliting the dataset.

04/12/2023
we rerun the model to update the metrics.
'''

import configs
from load_data_new import CVEDataset
# from load_data_colbert import CVEDataset #### 
import logging
from torch.utils.data import DataLoader
import os
import wandb



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
gpus = [0,1,2,3]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# set `no_deprecation_warning=True` to disable this warning




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

class CVEClassifier(pl.LightningModule):
    def __init__(self, 
                 num_classes=1,
                 dropout=0.1,
                 lr=5e-5,
                 num_train_epochs=20,
                 warmup_steps=1000,
                 ):
        
        super().__init__()
        self.codeReviewer = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/codereviewer").encoder
        
        self.save_hyperparameters()
        self.dropout = dropout
        self.criterion = nn.BCEWithLogitsLoss()

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # Fully connected layer for output
        self.fc = nn.Linear(2 * self.codeReviewer.config.hidden_size, num_classes)

    def forward(self, input_ids_desc, attention_mask_desc, input_ids_msg_diff, attention_mask_msg_diff):
        
        # Get [CLS] embeddings for desc and msg+diff
        desc_cls_embed = self.codeReviewer(input_ids=input_ids_desc, attention_mask=attention_mask_desc).last_hidden_state[:, 0, :]
        msg_diff_cls_embed = self.codeReviewer(input_ids=input_ids_msg_diff, attention_mask=attention_mask_msg_diff).last_hidden_state[:, 0, :]
        
        # Concatenate [CLS] embeddings
        concatenated = torch.cat((desc_cls_embed, msg_diff_cls_embed), dim=1)
        
        # Apply dropout
        dropped = self.dropout_layer(concatenated)
        
        # Pass through the fully connected layer
        output = self.fc(dropped)
        
        return output
    
    def common_step(self, batch):
        predict = self(
            batch['input_ids_desc'],
            batch['attention_mask_desc'],
            batch['input_ids_msg_diff'],  # Updated to msg_diff
            batch['attention_mask_msg_diff']  # Updated to msg_diff
        )
        predict = predict.squeeze(1)
        loss = self.criterion(predict, batch['label'])
        return loss

    def training_step(self, batch, dataloader_idx=None):
        loss = self.common_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self.common_step(batch)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self.common_step(batch)

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
if __name__ == '__main__':
    ####### Load the data loaders
    configs.get_singapore_time()
    logging.info('1/4: start to prepare the dataset.')

    train_data = CVEDataset(configs.train_file)
    valid_data = CVEDataset(configs.valid_file)
    test_data = CVEDataset(configs.test_file)

    train_dataloader = DataLoader(dataset=train_data, shuffle=True, batch_size=32, num_workers=15)
    valid_dataloader = DataLoader(dataset=valid_data, batch_size=32, num_workers=15)
    test_dataloader = DataLoader(dataset=test_data, batch_size=8, num_workers=15)

    batch = next(iter(train_dataloader))
    print(batch.keys())

    wandb.login()
    # 8f66cd17219a1912e8a14a65348e656c657f6c5e



    """Next, we initialize the model."""
    configs.get_singapore_time()
    ###### Load the model ######
    logging.info('2/4: start to construct our model.')


    model = CVEClassifier(
            num_classes=1,   # binary classification
            dropout=0.1
        )


    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
    import os


    # wandb_logger = WandbLogger(name='train_0831', project='PatchSleuth')
    # wandb_logger = WandbLogger(name='train_1004', project='PatchSleuth_ColBERT')
    # wandb_logger = WandbLogger(name='train_1007_20epoch', project='PatchSleuth_ColBERT_20epoch')
    wandb_logger = WandbLogger(name='train_1007_512', project='PatchSleuth_ColBERT')
    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=4,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # CHECK_POINTS_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/model/output_0831/Checkpoints"
    # CHECK_POINTS_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/model/output_1004/Checkpoints"
    # CHECK_POINTS_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/model/output_1007/Checkpoints"
    
    CHECK_POINTS_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/model/output_1007_512/Checkpoints"

    os.makedirs(CHECK_POINTS_PATH, exist_ok=True)


    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath=CHECK_POINTS_PATH,
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )


    trainer = Trainer(
        accelerator='gpu',                    # "cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto", "ddp"
        num_nodes=1,                          # Using 1 machine/node
        devices=len(gpus),                            # Using 4 GPUs
        default_root_dir=CHECK_POINTS_PATH,   # Default checkpoint path
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
    model_save_path = os.path.join(CHECK_POINTS_PATH, "final_model.pt")

    #### So we do not need to load the model Class when evaluate.
    torch.save(model, model_save_path)

    configs.get_singapore_time()


