import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set GPU and model details
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
base_model = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16
)
tokenizer.pad_token = tokenizer.eos_token

# Function to extract likelihood score from output
def extract_likelihood_score(output):
    output = output.strip().split('\n')[-1]
    return float(output.split(' ')[-1])

def process_batch(batch_prompts, model, tokenizer):
    model_inputs = tokenizer(batch_prompts, padding=True, return_tensors='pt', truncation=True, max_length=512)  # Adjust max_length if needed
    model_inputs = {k: v.to('cuda') for k, v in model_inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**model_inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.8)
    return [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]


# Load the dataset
DATA_DIR = '/mnt/local/Baselines_Bugs/PatchFinder/data'
SAVE_DIR = '/mnt/local/Baselines_Bugs/PatchFinder/codellama/retrieval/gpu_3'
os.makedirs(SAVE_DIR, exist_ok=True)

test_df_sampled = pd.read_csv(os.path.join(DATA_DIR, 'codellama_test_data_sampled_2.csv'))

# Batch processing setup
BATCH_SIZE = 8
results = []
batch_prompts = []
LAST_COMPLETED_IDX = 19649  # Last completed index

# Main processing loop
for idx, row in tqdm(test_df_sampled.iterrows(), total=test_df_sampled.shape[0]):

    if idx < LAST_COMPLETED_IDX:
        continue

    desc_tokens = row['desc_token'] 
    
    if isinstance(row['msg_token'], str):
        msg_tokens = row['msg_token'].split(' ')
        if len(msg_tokens) > 64:
            msg_tokens = msg_tokens[:64]
        
        msg_tokens = ' '.join(msg_tokens)
    else:
        msg_tokens = ' '

    if isinstance(row['diff_token'], str):
        diff_tokens = row['diff_token'].split(' ')
        if len(diff_tokens) > 128:
            diff_tokens = diff_tokens[:128]
        diff_tokens = ' '.join(diff_tokens)
    else:
        diff_tokens = ' '
    
    prompt = f"""\
[INST] You are an expert on security patches, evaluate the likelihood that the provided code commit is a patch \
for the given CVE description.\
provide a score from 0 to 100, where 0 means not a patch and 100 means definitely a patch.\n\
CVE description: {desc_tokens}\n\
commit msg: {msg_tokens}\
diff: {diff_tokens}\n\
Your answer must only contain the numerical score, with no other text or symbols.
[/INST]
    """

    batch_prompts.append(prompt)

    if len(batch_prompts) == BATCH_SIZE:
        batch_outputs = process_batch(batch_prompts, model, tokenizer)
        for output_idx, output in enumerate(batch_outputs):
            save_idx = idx - BATCH_SIZE + 1 + output_idx  # Calculate the correct original index
            with open(os.path.join(SAVE_DIR, f'{save_idx}.txt'), 'w') as f:
                f.write(output + '\n')
        batch_prompts = []

# Processing the last batch (if any)
if batch_prompts:
    batch_outputs = process_batch(batch_prompts, model, tokenizer)
    for output_idx, output in enumerate(batch_outputs):
        save_idx = len(test_df_sampled) - len(batch_prompts) + output_idx  # Calculate the correct original index
        with open(os.path.join(SAVE_DIR, f'{save_idx}.txt'), 'w') as f:
            f.write(output + '\n')

print('Inference completed and outputs saved.')