import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df):
        self.dataframe = df
        # self.tokenizer = tokenizer
        self.msg_tokens = self.dataframe['msg_token']
        self.diff_tokens = self.dataframe['diff_token']

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        msg_token = self.msg_tokens[idx] if isinstance(self.msg_tokens[idx], str) else ' '
        diff_token = self.diff_tokens[idx] if isinstance(self.diff_tokens[idx], str) else ' '

        return {
            'msg_token': msg_token,
            'diff_token': diff_token
        }