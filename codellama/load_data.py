import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        msg_token = row['msg_token'] if isinstance(row['msg_token'], str) else ' '
        diff_token = row['diff_token'] if isinstance(row['diff_token'], str) else ' '
        commit_token = msg_token + ' ' + diff_token
        
        # encoding = self.tokenizer(f"{msg_token} {diff_token}", return_tensors='pt', truncation=True, padding=True)
        encoding = self.tokenizer(
            commit_token,
            return_tensors='pt',
            truncation=True,
            padding=True)
        
        return encoding