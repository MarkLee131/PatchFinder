import os
import torch
# from torchtext.legacy import data #### torchtext0.9.0 should use this
from torchtext import data
import configs

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")

SEED = 3407

class CustomDataset(data.Dataset):
    def __init__(self, examples, fields, **kwargs):
        super(CustomDataset, self).__init__(examples, fields, **kwargs)


#### need to return a list for each desc, msg, and diff.

#### for code diff
## input: tokenized code diff by using regex
## output: tokenized code diff by further using codereviewer
def diff_tokenize(diff_tokenized:str):
    ### we should use codereviewer to tokenize the code diff.
    
    ''' the following line (encode) not only tokenizes the input but also converts each token into an ID according to 
    the tokenizer's vocabulary. 
    This is useful because the model needs these IDs to understand the tokens. 
    # diff_tokens = tokenizer.encode(diff_tokenized, truncation=True, max_length=512)
    If you just want the tokens and don't need them to be converted into IDs, you can use tokenizer.tokenize instead.
    '''
    diff_tokens = tokenizer.tokenize(diff_tokenized)
    return diff_tokens



#### for commit message
def msg_tokenize(msg:str):
    # msg_tokens = msg.split(' ', maxsplit=511)
    msg_tokens = tokenizer.tokenize(msg)
    ### for data will be input into LSTM, we should truncate the length of the code to 512 or less. len(list)
    return msg_tokens[:512]


#### for CVE description
def desc_tokenize(cve_desc:str):
    desc_tokens = cve_desc.split(' ', maxsplit=511)
    return desc_tokens


#### tokenize the data...
def prepare_data():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    DIFF = data.Field(tokenize = diff_tokenize,
                    include_lengths=True,batch_first=True, fixed_length=512) ##### for code, we should truncate the length of the code to 512 or less. len(list)
    
    ##### TODO to determine whether we should use fixed_length or not, we should check the codereviewer usage.
    
    MSG = data.Field(tokenize = msg_tokenize,
                    include_lengths=True,batch_first=True)
    
    DESC = data.Field(tokenize = desc_tokenize,
                    include_lengths=True,batch_first=True)
    
    
    ##### leave it as default
    LABEL = data.LabelField(dtype = torch.float,batch_first=True)

    if not configs.save_dataset:
        fields = [("diff_token", DIFF),("msg_token", MSG),('desc_token', DESC),("label",LABEL)]
        
        #### Caution: take care this field, will load all data.
        train_data, valid_data, test_data = data.TabularDataset.splits(
            path=configs.data_path,
            train=configs.train_file,
            validation=configs.valid_file,
            test=configs.test_file,
            format='csv',
            fields=fields,
            skip_header=True
        )
    
        ####### tokenize end here #########


        ##### build vocabulary ---------
        MAX_VOCAB_SIZE = 100000
        DIFF.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        MSG.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        DESC.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        
        
        torch.save(DIFF, os.path.join(configs.save_path,'field_diff.pth'))
        torch.save(MSG, os.path.join(configs.save_path,'field_msg.pth'))
        torch.save(DESC, os.path.join(configs.save_path,'field_desc.pth'))
        torch.save(LABEL, os.path.join(configs.save_path,'field_label.pth'))
        
        torch.save(train_data.examples, os.path.join(configs.save_path,'train_data.pth'))
        torch.save(valid_data.examples, os.path.join(configs.save_path,'valid_data.pth'))
        torch.save(test_data.examples, os.path.join(configs.save_path,'test_data.pth'))
        
    else:

        train_data_examples = torch.load(os.path.join(configs.save_path,'train_data.pth'))
        valid_data_examples = torch.load(os.path.join(configs.save_path,'valid_data.pth'))
        test_data_examples = torch.load(os.path.join(configs.save_path,'test_data.pth'))
        DIFF=torch.load(os.path.join(configs.save_path,'field_diff.pth'))
        MSG=torch.load(os.path.join(configs.save_path,'field_msg.pth'))
        DESC=torch.load(os.path.join(configs.save_path,'field_desc.pth'))
        LABEL=torch.load(os.path.join(configs.save_path,'field_label.pth'))
        fields = [("Diff", DIFF),("Msg", MSG),('Desc',DESC),("Label",LABEL)]

        train_data = CustomDataset(train_data_examples, fields)
        valid_data = CustomDataset(valid_data_examples, fields)
        test_data = CustomDataset(test_data_examples, fields)
        MAX_VOCAB_SIZE = 100000
        DIFF.build_vocab(train_data.Diff, max_size = MAX_VOCAB_SIZE)
        MSG.build_vocab(train_data.Msg, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        DESC.build_vocab(train_data.Desc, max_size = MAX_VOCAB_SIZE)
        ###################TODO Ask Han, what meaning for each parameter in build_vocab
        
        
    ##### construct iterator
    device = configs.device
    BATCH_SIZE = configs.batch_size
    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.diff),
                                device=device, sort=True, sort_within_batch=True)
    valid_iter = data.BucketIterator(valid_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.diff),
                                device=device, sort=True, sort_within_batch=True)
    test_iter = data.BucketIterator(test_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.diff),
                                device=device, sort=True, sort_within_batch=True)

    return train_iter, valid_iter, test_iter
