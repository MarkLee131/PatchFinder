import os
import torch
from torchtext.legacy import data
import configs
from antlr4 import *
import nltk
import ast
nltk.download('punkt')
from nltk.tokenize import word_tokenize
SEED = 3407

class CustomDataset(data.Dataset):
    def __init__(self, examples, fields, **kwargs):
        super(CustomDataset, self).__init__(examples, fields, **kwargs)


#### need to return a list for each desc, msg, and diff.
def code_token():
    return


def msg_token():
### for data will be input into LSTM, we should truncate the length of the code to 512 or less. len(list)
    return 



#### tokenize the data...
def prepare_data():
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    #javalang token
    CODE = data.Field(tokenize = code_tokenize,
                    include_lengths=True,batch_first=True, fixed_length=512) ##### for code, we should truncate the length of the code to 512 or less. len(list)
    ##### to determine whether we should use fixed_length or not, we should check the codereviewer usage.
    
    
    MSG = data.Field(tokenize = msg_tokenize,
                    include_lengths=True,batch_first=True)
    
    FIELD=data.Field(tokenize = field_tokenize,
                    include_lengths=True,batch_first=True)
    
    FIELDSWARNING = data.Field(tokenize = field_warning_tokenize,
                    batch_first=True)
    
    #### if the data is some float or int, we use the follows, but in this work, we do not need 
    PRIORITY = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)
    RANK = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)      
    RULE = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)
    CAT = data.Field(sequential=False, use_vocab=False,dtype=torch.long,batch_first=True)
    
    
    ##### leave it as default
    LABEL = data.LabelField(dtype = torch.float,batch_first=True)

    if not configs.save_dataset:
        fields = [("Code", CODE),("Msg", MSG),('Field',FIELD),("Cat", CAT),("Rule",RULE),("Priority",PRIORITY),("Rank",RANK),('FieldWarning',FIELDSWARNING),("Label",LABEL)]
        
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
        CODE.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        MSG.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        FIELDSWARNING.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        FIELD.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        
        
        torch.save(CODE, os.path.join(configs.save_path,'field_code.pth'))
        torch.save(MSG, os.path.join(configs.save_path,'field_msg.pth'))
        torch.save(RULE, os.path.join(configs.save_path,'field_rule.pth'))
        torch.save(CAT, os.path.join(configs.save_path,'field_cat.pth'))
        torch.save(FIELD, os.path.join(configs.save_path,'field_field.pth'))
        torch.save(FIELDSWARNING, os.path.join(configs.save_path,'field_fieldwarning.pth'))
        torch.save(PRIORITY, os.path.join(configs.save_path,'field_priority.pth'))
        torch.save(RANK, os.path.join(configs.save_path,'field_rank.pth'))
        torch.save(LABEL, os.path.join(configs.save_path,'field_label.pth'))
        torch.save(train_data.examples, os.path.join(configs.save_path,'train_data.pth'))
        torch.save(valid_data.examples, os.path.join(configs.save_path,'valid_data.pth'))
        torch.save(test_data.examples, os.path.join(configs.save_path,'test_data.pth'))
    else:

        train_data_examples = torch.load(os.path.join(configs.save_path,'train_data.pth'))
        valid_data_examples = torch.load(os.path.join(configs.save_path,'valid_data.pth'))
        test_data_examples = torch.load(os.path.join(configs.save_path,'test_data.pth'))
        CODE=torch.load(os.path.join(configs.save_path,'field_code.pth'))
        MSG=torch.load(os.path.join(configs.save_path,'field_msg.pth'))
        RULE=torch.load(os.path.join(configs.save_path,'field_rule.pth'))
        CAT=torch.load(os.path.join(configs.save_path,'field_cat.pth'))
        FIELD=torch.load(os.path.join(configs.save_path,'field_field.pth'))
        FIELDSWARNING=torch.load(os.path.join(configs.save_path,'field_fieldwarning.pth'))
        PRIORITY=torch.load(os.path.join(configs.save_path,'field_priority.pth'))
        RANK=torch.load(os.path.join(configs.save_path,'field_rank.pth'))
        LABEL=torch.load(os.path.join(configs.save_path,'field_label.pth'))
        fields = [("Code", CODE),("Msg", MSG),('Field',FIELD),("Cat", CAT),("Rule",RULE),("Priority",PRIORITY),("Rank",RANK),('FieldWarning',FIELDSWARNING),("Label",LABEL)]

        train_data = CustomDataset(train_data_examples, fields)
        valid_data = CustomDataset(valid_data_examples, fields)
        test_data = CustomDataset(test_data_examples, fields)
        MAX_VOCAB_SIZE = 100000
        CODE.build_vocab(train_data.Code,train_data.Field, max_size = MAX_VOCAB_SIZE)
        MSG.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        FIELDSWARNING.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        FIELD.build_vocab(train_data.Code,train_data.Field, max_size = MAX_VOCAB_SIZE)
        # FIELD.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
        
        
    ##### construct iterator
    device = configs.device
    BATCH_SIZE = configs.batch_size
    train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Code),
                                device=device, sort=True, sort_within_batch=True)
    valid_iter = data.BucketIterator(valid_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Code),
                                device=device, sort=True, sort_within_batch=True)
    test_iter = data.BucketIterator(test_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.Code),
                                device=device, sort=True, sort_within_batch=True)

    return train_iter, valid_iter, test_iter
