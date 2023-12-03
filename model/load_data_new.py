import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
'''
concat the msg and diff tokens to train
'''
class CVEDataset(Dataset):
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)
        self.cve = self.df['cve']
        self.desc_tokens = self.df['desc_token']
        # Combine msg and diff tokens with a space separator
        self.msg_diff_tokens = self.df['msg_token'] + " " + self.df['diff_token']
        self.labels = self.df['label']
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
        
    def __getitem__(self, index):
        desc = self.desc_tokens[index] if isinstance(self.desc_tokens[index], str) else ''
        desc_encoding = self.tokenizer.encode_plus(
            desc,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        msg_diff = self.msg_diff_tokens[index] if isinstance(self.msg_diff_tokens[index], str) else ''
        msg_diff_encoding = self.tokenizer.encode_plus(
            msg_diff,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids_desc': desc_encoding['input_ids'].flatten(),
            'attention_mask_desc': desc_encoding['attention_mask'].flatten(),
            'input_ids_msg_diff': msg_diff_encoding['input_ids'].flatten(),
            'attention_mask_msg_diff': msg_diff_encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[index], dtype=torch.float),
            'cve': self.cve[index]
        }

    def __len__(self):
        return len(self.df)
