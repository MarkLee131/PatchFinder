import os
import pandas
import convert_rule2num
import pandas as pd
import random
import re
file='/wkspace/Predict/data/negative.csv'
df = pd.read_csv(file)
result=[]
for index, row in df.iterrows():
    temp=[]
    temp.append(re.sub(' +', ' ',row['bug_code'].replace('\n','').replace('\t','')))
    temp.append(row['msg'])
    temp.append(convert_rule2num.Rule2Num(row['rule']))
    temp.append(convert_rule2num.Cat2Num(row['cat']))
    temp.append('pos')
    result.append(temp)
file='/wkspace/Predict/data/positive.csv'
df = pd.read_csv(file)
for index, row in df.iterrows():
    temp=[]
    temp.append(re.sub(' +', ' ',row['bug_code'].replace('\n','').replace('\t','')))
    temp.append(row['msg'])
    temp.append(convert_rule2num.Rule2Num(row['rule']))
    temp.append(convert_rule2num.Cat2Num(row['cat']))
    temp.append('neg')
    result.append(temp)    
random.shuffle(result)
df = pd.DataFrame(result,columns=['Code','Msg','Rule','Cat','Label'])
df.to_csv('/wkspace/Predict/data/warnings.csv',index=False)