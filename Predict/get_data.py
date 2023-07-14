import pandas as pd
import numpy as np
import pymongo
from tqdm import tqdm
import logging
import random
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def get_data():
    random.seed(3407)
    mongo='mongodb://155.69.147.181:28888/'
    db=pymongo.MongoClient(mongo)
    collection1=db['bugs']['warnings1']
    warnings=collection1.find({'cf_df_code1':{'$exists':True}},
                            {'cf_df_code1':1,'msg':1,'Cat':1,'Type':1,'is_pos':1,'_id':1,'field_code':1,'Priority':1,'Rank':1,'field':1})
    result=[]
    for warning in tqdm(warnings):
        try:
            if warning['cf_df_code1']=='' or warning['cf_df_code1']==None or type(warning['cf_df_code1'])!=str or warning['cf_df_code1']==np.nan:
                continue
            else:
                bug_code=warning['cf_df_code1']
                bug_code=bug_code.strip()
            # if '\0' in bug_code:
            #     print(bug_code)
            #     continue
            # bug_code=bug_code.replace('\0','')
            if warning['msg']=='' or warning['msg']==None or warning['msg']==np.nan:
                continue
            else:
                msg=warning['msg']
            
            if warning['field_code']=='' or warning['field_code']==None:
                field_code='No Field Code'
            else:
                field_code=warning['field_code']
            result.append({'Code':bug_code,'Msg':msg,'FieldCode':field_code,'Cat':Cat2Num(warning['Cat']),'Rule':Rule2Num(warning['Type']),
                        'Priority':int(warning['Priority']),'Rank':int(warning['Rank']),'field':warning['field'],'Label':'pos' if 'is_pos' in warning and warning['is_pos']==True else 'neg'})
        except Exception as e:
            print(e)


    # Assuming your list of dictionaries is called list_of_dicts
    result = [json.loads(t) for t in set([json.dumps(d) for d in result])]

    print(len(result))
    data=pd.DataFrame(result)
    data.to_csv('/mnt/local/Predict5/data/warnings1.csv',index=False)
    X = data.drop('Label', axis=1)  # Replace 'label_column' with the column name of your labels
    y = data['Label']               # Replace 'label_column' with the column name of your labels

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=3407)

    # Split the remaining data into validation and test sets (10% validation, 10% test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3407)

    # Now you have your train, validation, and test sets
    # Combine the features and labels for each set
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_val, y_val], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save the sets as CSV files
    train_data.to_csv('/mnt/local/Predict5/data/train.csv', index=False)
    val_data.to_csv('/mnt/local/Predict5/data/valid.csv', index=False)
    test_data.to_csv('/mnt/local/Predict5/data/test.csv', index=False)
    return train_data,val_data,test_data