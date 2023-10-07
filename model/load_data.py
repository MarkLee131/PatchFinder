import pandas as pd
import torch
from torch.utils.data import Dataset
# Load model directly
from transformers import AutoTokenizer

'''
train model by not concat the msg and diff tokens, tokenized separately
'''
class CVEDataset(Dataset):
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)
        self.cve = self.df['cve']
        self.desc_tokens = self.df['desc_token']
        self.msg_tokens = self.df['msg_token']
        self.diff_tokens = self.df['diff_token']
        self.labels = self.df['label']
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
        
    def __getitem__(self, index):
        desc=self.desc_tokens[index] if isinstance(self.desc_tokens[index],str) else ''
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
        msg = self.msg_tokens[index] if isinstance(self.msg_tokens[index],str) else ''
        msg_encoding = self.tokenizer.encode_plus(
        msg,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
        )
        
        diff = self.diff_tokens[index] if isinstance(self.diff_tokens[index],str) else ''
        diff_encoding = self.tokenizer.encode_plus(
        diff,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
        )        

        return {
            'input_ids_desc': desc_encoding['input_ids'].flatten(),
            'attention_mask_desc': desc_encoding['attention_mask'].flatten(),
            'input_ids_msg': msg_encoding['input_ids'].flatten(),
            'attention_mask_msg': msg_encoding['attention_mask'].flatten(),
            'input_ids_diff': diff_encoding['input_ids'].flatten(),
            'attention_mask_diff': diff_encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[index], dtype=torch.float),
            'cve': self.cve[index]
        }

    def __len__(self):
        return len(self.df)



# ### ChatGPT prompt:
# You are an AI programmer for software engineering tasks. 
# now we try to train a model to locate the patch commits for a give CVE description. 
# We expect to use bi-LSTM and the pretrained model Codereviewer to train a model to solve it.
# For the load_data.py, we have csv file, each row includes cve, desc_token(tokenized CVE description), 
# msg_token(tokenized commit message), diff_token(tokenized code diff), label(0 or 1).
# We want to further finetune the pretrained model Codereviewer to tokenize the commit msg, and code diff, 
# and concat the tokenized data: desc_token, msg_token, diff_token, and then feed the data into bi-LSTM.