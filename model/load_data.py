import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# Load model directly
from transformers import AutoTokenizer
import configs

class CSVDataset(Dataset):
    def __init__(self, desc_tokens, msg_tokens, diff_tokens, labels):
        self.desc_tokens = desc_tokens
        self.msg_tokens = msg_tokens
        self.diff_tokens = diff_tokens
        self.labels = labels

    def __len__(self):
        return len(self.desc_tokens)

    def __getitem__(self, idx):
        return self.desc_tokens[idx], self.msg_tokens[idx], self.diff_tokens[idx], self.labels[idx]

def load_data(file):
    data = pd.read_csv(file)

    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
    # model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")

    desc_tokens = data['desc_token'].tolist()
    msg_tokens = [code_tokenizer.tokenize(msg) for msg in data['msg_token'].tolist()]
    diff_tokens = [code_tokenizer.tokenize(diff) for diff in data['diff_token'].tolist()]
    labels = data['label'].tolist()

    dataset = CSVDataset(desc_tokens, msg_tokens, diff_tokens, labels)

    return DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

if __name__ == '__main__':
    train_loader = load_data(configs.data_path + "/" + configs.train_file)
    valid_loader = load_data(configs.data_path + "/" + configs.valid_file)
    test_loader = load_data(configs.data_path + "/" + configs.test_file)



### ChatGPT prompt:
# You are an AI programmer for software engineering tasks. 
# now we try to train a model to locate the patch commits for a give CVE description. 
# We expect to use bi-LSTM and the pretrained model Codereviewer to train a model to solve it.
# For the load_data.py, we have csv file, each row includes cve, desc_token(tokenized CVE description), 
# msg_token(tokenized commit message), diff_token(tokenized code diff), label(0 or 1).
# We want to further finetune the pretrained model Codereviewer to tokenize the commit msg, and code diff, 
# and concat the tokenized data: desc_token, msg_token, diff_token, and then feed the data into bi-LSTM.