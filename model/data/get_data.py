import pandas as pd
import numpy as np
import pymongo
from convert_rule2num import Rule2Num,Cat2Num
from tqdm import tqdm
import logging
import random
import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import multiprocessing
random.seed(3407)
mongo='mongodb://155.69.148.184:28888/'
db=pymongo.MongoClient(mongo)
collection1=db['bugs']['warnings1']
warnings=collection1.find({'unique':True},
                          {'cf_df_code1':1,'msg':1,'Cat':1,'Type':1,'is_pos':1,'_id':1,'field_code':1,'Priority':1,'Rank':1,'field':1,'bug_code':1})
result=[]
cou=0
for warning in tqdm(warnings):
    try:
        if 'cf_df_code1' not in warning:
            continue
        if warning['cf_df_code1']=='' or warning['cf_df_code1']==None or type(warning['cf_df_code1'])!=str or warning['cf_df_code1']==np.nan:
            continue
        else:
            bug_code=warning['cf_df_code1']
            bug_code=bug_code.strip()

        if warning['msg']=='' or warning['msg']==None or warning['msg']==np.nan:
            continue
        else:
            msg=warning['msg']
        
        if warning['field_code']=='' or warning['field_code']==None:
            field_code='NoFieldCode'
        else:
            field_code=warning['field_code']
        result.append({'Code':bug_code,'Msg':msg,'FieldCode':field_code,'Cat':Cat2Num(warning['Cat']),'Rule':Rule2Num(warning['Type']),
                       'Priority':int(warning['Priority']),'Rank':int(warning['Rank']),'field':warning['field'],'Label':'pos' if 'is_pos' in warning and warning['is_pos']==True else 'neg'})
    except Exception as e:
        print(e)


final_result=[]
for i in result:
    i['Code']=i['Code'].encode('utf-8')
    final_result.append(i)
print(len(result))
data=pd.DataFrame(final_result)
data.to_csv('/mnt/local/Predict5/data1/warnings.csv',index=False)
