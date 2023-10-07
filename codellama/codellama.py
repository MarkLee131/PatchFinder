'''
We try to use CodeLlama-34b-Instruct-hf model to narrow down the search space for the patch commits.
'''

# Import necessary libraries
import os
import pandas as pd
import torch
from tqdm import tqdm
from configs import *
from load_data import CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# Load the dataset
DATA_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/data'
SAVE_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/codellama/retrieval'
os.makedirs(SAVE_DIR, exist_ok=True)

# Define infer_patches function
def infer_patches(dataloader):
    model.eval()  # Set the model to evaluation mode
    all_probs = []
    with torch.no_grad():  # Disable gradient calculations
        for batch in tqdm(dataloader, desc="Inferencing"):  # Wrap dataloader with tqdm for progress bar
            inputs = {k: v.to(device) for k, v in batch.items()}  # Move inputs to the device
            outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)[:, 1].cpu().numpy()  # Get 'patch' probabilities
            all_probs.extend(probs)
    return all_probs




# Define select_top_commits function
def select_top_commits(df, probs):
    df['patch_probability'] = probs  # Add probabilities to DataFrame
    top_commits = df.groupby('cve').apply(lambda x: x.nlargest(100, 'patch_probability')).reset_index(drop=True)
    return top_commits


if __name__ == "__main__":
    # train_file = os.path.join(DATA_DIR, 'train_data.csv')
    # validate_file = os.path.join(DATA_DIR, 'validate_data.csv')
    test_file = os.path.join(DATA_DIR, 'test_data.csv')
    
    get_singapore_time()
    print('Loading data...')
    
    # train_df = pd.read_csv(train_file)
    # validate_df = pd.read_csv(validate_file)
    test_df = pd.read_csv(test_file)
    
    get_singapore_time()
    print('Data loaded.')
    
    get_singapore_time()
    print('Creating DataLoaders...')
    # # Create DataLoaders
    # train_dataset = CustomDataset(train_df, tokenizer)
    # train_dataloader = DataLoader(train_dataset, batch_size=32)  # Adjust batch size as needed

    # validate_dataset = CustomDataset(validate_df, tokenizer)
    # validate_dataloader = DataLoader(validate_dataset, batch_size=32)  # Adjust batch size as needed

    test_dataset = CustomDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=32)  # Adjust batch size as needed
    
    
    # Infer patches
    
    get_singapore_time()
    print('Inferencing...')
    # train_probs = infer_patches(train_dataloader)
    # validate_probs = infer_patches(validate_dataloader)
    test_probs = infer_patches(test_dataloader)
    
    
    
    # Select top 200 commits for each CVE
    get_singapore_time()
    print('Selecting top commits...')
    
    
    # top_train_commits = select_top_commits(train_df, train_probs)
    # top_train_commits.to_csv(os.path.join(SAVE_DIR, 'top_train_commits.csv'), index=False)

    # top_validate_commits = select_top_commits(validate_df, validate_probs)
    # top_validate_commits.to_csv(os.path.join(SAVE_DIR, 'top_validate_commits.csv'), index=False)

    top_test_commits = select_top_commits(test_df, test_probs)
    top_test_commits.to_csv(os.path.join(SAVE_DIR, 'top_test_commits.csv'), index=False)
    
    
