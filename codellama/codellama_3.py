import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
base_model = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16
)

# Load the dataset
DATA_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/data'
SAVE_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/codellama/retrieval/gpu_3'
os.makedirs(SAVE_DIR, exist_ok=True)


print('Loading data...')

test_df_sampled = pd.read_csv(os.path.join(DATA_DIR, 'codellama_test_data_sampled_2.csv'))


print('Data loaded...')

def extract_likelihood_score(output):
    ### keep the last line of the output
    output = output.strip().split('\n')[-1]
    ### remove the prompt
    return float(output.split(' ')[-1])
    

print('Inferencing...')
results = []

for idx, row in tqdm(test_df_sampled.iterrows(), total=test_df_sampled.shape[0]):
    desc_tokens = row['desc_token'] 
    
    if isinstance(row['desc_token'], str):
        msg_tokens = row['msg_token'].split(' ')
        if len(msg_tokens) > 64:
            msg_tokens = msg_tokens[:64]
        
        msg_tokens = ' '.join(msg_tokens)
    else:
        msg_tokens = ' '

    if isinstance(row['diff_token'], str):
        diff_tokens = row['diff_token'].split(' ')
        if len(diff_tokens) > 512:
            diff_tokens = diff_tokens[:512]    
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

    model_input = tokenizer(prompt, return_tensors='pt', max_length=1800, truncation=True)
    model_input = model_input.to('cuda')
    with torch.no_grad():
        try:
            output = model.generate(**model_input, num_return_sequences=1, max_new_tokens=1024, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.8)[0]
            decoded_output = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            prompt_end_index = decoded_output.find("[/INST]") + len("[/INST]")
            generated_output = decoded_output[prompt_end_index:].strip()
        except Exception as e:
            print(e)
            decoded_output = e
            continue
            
    # print(decoded_output)
    with open(os.path.join(SAVE_DIR, f'{idx}.txt'), 'w') as f:
        f.write(decoded_output + '\n')
            
    try:
        likelihood_score = extract_likelihood_score(decoded_output)
        # print(likelihood_score)        
        results.append(likelihood_score)
    except Exception as e:
        print(e)
        results.append(-1)
        continue

# Add results to DataFrame and save to CSV
test_df_sampled['likelihood'] = results
test_df_sampled.sort_values(by='likelihood', ascending=False, inplace=True)
test_df_sampled.to_csv(os.path.join(SAVE_DIR, 'inference_results.csv'), index=False)

print('Inference completed and results saved.')
