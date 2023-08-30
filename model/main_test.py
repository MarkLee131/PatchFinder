from datasets import load_dataset

dataset = load_dataset("code_x_glue_ct_code_to_text", "ruby")
# print(dataset)

"""As you can see, the "code-to-text/ruby" split consists of a training, validation and test set. Let's look at one particular example:"""

example = dataset['train'][0]

# print("Code:", example["code"])
# print("Docstring:", example["docstring"])

"""The goal for the model is to generate a docstring based on the provided code.

Let's now prepare the examples (i.e. code-docstring pairs) for the model. As you might know, Transformer models like BERT, BART, T5 etc. don't expect text as direct input, but rather integers which are called `input_ids` in HuggingFace Transformers. These represent tokens of a certain vocabulary. The model will learn rich contextual embedding vectors for each token, allowing it to get good results.

In other words, we need to turn the "Code" input from above into `input_ids`, and similarly, we need to turn the "Docstring" output from above into `input_ids`, which will serve as the `labels` for the model.

In addition, as these models are trained on batches of examples rather than one example at a time, we'll need to pad/truncate both the inputs and labels, such that they are all of the same length. That's why we also will add an `attention_mask` input to the model, such that it knows not to take into account padding tokens when computing attention scores.

To summarize:
* input: code, which is turned into `input_ids` + `attention_mask`
* output: docstrings, which are turned into `labels` (which are the `input_ids` of the docstrings).

Below, we define a `preprocess_examples` function, which we can apply on the entire dataset.
"""

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

prefix = "Summarize Ruby: "
max_input_length = 256
max_target_length = 128

def preprocess_examples(examples):
  # encode the code-docstring pairs
  codes = examples['code']
  docstrings = examples['docstring']

  inputs = [prefix + code for code in codes]
  model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

  # encode the summaries
  labels = tokenizer(docstrings, max_length=max_target_length, padding="max_length", truncation=True).input_ids

  # important: we need to replace the index of the padding tokens by -100
  # such that they are not taken into account by the CrossEntropyLoss
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)

  model_inputs["labels"] = labels_with_ignore_index

  return model_inputs

"""Now that we have defined the function, let's call `.map()` on the HuggingFace Dataset object, which allows us to apply this function in batches (by default a batch size of 1,000 is used!) - hence super fast."""

dataset = dataset.map(preprocess_examples, batched=True)

# print(dataset)

"""Next, let's set the format to "torch" and create PyTorch dataloaders."""

from torch.utils.data import DataLoader

dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8, num_workers=4)
valid_dataloader = DataLoader(dataset['validation'], batch_size=4, num_workers=4)
test_dataloader = DataLoader(dataset['test'], batch_size=4, num_workers=4)

batch = next(iter(train_dataloader))
print(batch.keys())

"""Let's verify an example, by decoding it back into text:"""

tokenizer.decode(batch['input_ids'][0])

labels = batch['labels'][0]
tokenizer.decode([label for label in labels if label != -100])

"""## Fine-tune using PyTorch Lightning

As we will train the model using PyTorch Lightning, we first need to define a `LightningModule`, which is an `nn.Module` with some additional functionalities. We just need to define the `forward` pass, `training_step` (and optionally `validation_step` and `test_step`), and the corresponding dataloaders. PyTorch Lightning will then automate the training for us, handling device placement (i.e. we don't need to type `.to(device)` anywhere), etc. It also comes with support for loggers (such as Tensorboard, Weights and Biases) and callbacks.

Of course, you could also train the model in other ways:
* using regular PyTorch
* using the HuggingFace Trainer (in this case, the Seq2SeqTrainer)
* using HuggingFace Accelerate
* etc.
"""

from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl

class CodeT5(pl.LightningModule):
    def __init__(self, lr=5e-5, num_train_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

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

model = CodeT5()


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
gpus = [0,1,2,3]


wandb_logger = WandbLogger(name='test', project='CodeT5')
# for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
early_stop_callback = EarlyStopping(
    monitor='validation_loss',
    patience=3,
    strict=False,
    verbose=False,
    mode='min'
)
lr_monitor = LearningRateMonitor(logging_interval='step')

# trainer = Trainer(
#     default_root_dir="/content/drive/MyDrive/CodeT5/Notebooks/Checkpoints",
#     logger=wandb_logger,
#     callbacks=[early_stop_callback, lr_monitor],
#                   )
os.makedirs("/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints", exist_ok=True)

trainer = Trainer(
    accelerator='gpu',                    # Since we're using multiple GPUs, using distributed data parallel
    num_nodes=1,                          # Using 1 machine/node
    devices=4,                            # Using 4 GPUs
    default_root_dir="/mnt/local/Baselines_Bugs/PatchSleuth/output/Checkpoints",
    logger=wandb_logger,
    callbacks=[early_stop_callback, lr_monitor],
    max_epochs=20,                        # Set the maximum number of epochs
    accumulate_grad_batches=1,            # Gradient accumulation steps
    max_steps=100000,                     # Set the maximum number of training steps
    log_every_n_steps=100,                # Log every 100 steps
    precision=32,                         # Using 32-bit precision for training; this is the default and can be omitted if desired
    gradient_clip_val=0.0,                # Assuming you don't want gradient clipping, but adjust if needed
)


trainer.fit(model)

"""Once we're done training, we can also save the HuggingFace model as follows:"""

save_directory = "." # save in the current working directory, you can change this of course
model.model.save_pretrained(save_directory)

"""This allows us to easily load the trained model again using the `from_pretrained()` method, as shown below.

## Inference

Now that we've trained a model, let's test it on some examples from the test set.
"""

from datasets import load_dataset

dataset = load_dataset("code_x_glue_ct_code_to_text", "ruby")
print(dataset['test'])

test_example = dataset['test'][2]
print("Code:", test_example['code'])

"""We can load our trained model as follows:"""

from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(save_directory)

"""We can prepare the example using `RobertaTokenizer`, and generate using the `.generate()` method. Note that there are several ways of doing generation (greedy decoding/beam search/top k sampling/etc.), for that I refer to Patrick's blog post which you can find [here](https://huggingface.co/blog/how-to-generate). Here we will just use the default settings (i.e. greedy decoding)."""

# prepare for the model
input_ids = tokenizer(test_example['code'], return_tensors='pt').input_ids
# generate
outputs = model.generate(input_ids)
print("Generated docstring:", tokenizer.decode(outputs[0], skip_special_tokens=True))

"""Let's compare this to the ground-truth docstring:"""

print("Ground truth:", test_example['docstring'])
