'''
We try to use CodeLlama-34b-Instruct-hf model to narrow down the search space for the patch commits.
'''

# Import necessary libraries
import os
import pandas as pd
import torch
from tqdm import tqdm
# from configs import *
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
# model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

tokenizer = AutoTokenizer.from_pretrained("Riiid/sheep-duck-llama-2-70b-v1.1")
# model = AutoModelForCausalLM.from_pretrained("Riiid/sheep-duck-llama-2-70b-v1.1")

base_model = "Riiid/sheep-duck-llama-2-70b-v1.1"

# base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# model = model.to(device)

# Load the dataset
DATA_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/data'
SAVE_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/codellama/retrieval'
os.makedirs(SAVE_DIR, exist_ok=True)

if __name__ == "__main__":

    # train_file = os.path.join(DATA_DIR, 'train_data.csv')
    # validate_file = os.path.join(DATA_DIR, 'validate_data.csv')
    test_file = os.path.join(DATA_DIR, 'test_data_demo.csv')
    
    # get_singapore_time()
    print('Loading data...')
    
    # train_df = pd.read_csv(train_file)
    # validate_df = pd.read_csv(validate_file)
    test_df = pd.read_csv(test_file)
    
    # get_singapore_time()
    print('Data loaded (for head of test_df)...')
    
    # get_singapore_time()
    print('Inferencing...')
    
    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        msg_tokens = row['msg_token'] if isinstance(row['msg_token'], str) else ' '
        diff_tokens = row['diff_token'] if isinstance(row['diff_token'], str) else ' '
        # commits = row['msg_token'] + row['diff_token']
    
        prompt = (
            "Assume that you are a premier security expert tasked with locating patch commits "
            "for software vulnerabilities.\n"
            "You are presented with a code commit which includes a commit message and a code diff.\n"
            "Your task is to determine whether this commit is a patch for a known vulnerability.\n"
            "Let's break down the steps:\n"
            "1. Examine the commit message and the code diff to understand the changes made.\n"
            "2. Analyze whether the commit addresses a vulnerability/bug and is intended as a patch.\n"
            "3. Provide your final decision on whether this is a patch commit (a boolean value, i.e., True/False).\n"
            "### Input:"
            f"Commit message: {msg_tokens}\n"
            f"Code diff: {diff_tokens}\n"
            "### Response: "
        )
        print(prompt)
    
    
        model_input = tokenizer(prompt, return_tensors="pt").to('cuda')

        with torch.no_grad():
            print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True, pad_token_id=tokenizer.eos_token_id))
            # save the result to a file
            with open(os.path.join(SAVE_DIR, f'{idx}.txt'), 'w') as f:
                f.write(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True, pad_token_id=tokenizer.eos_token_id))