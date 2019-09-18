import sys
import numpy as np
import pickle as pc
import os
sys.path.append('/media/sda1/nana/mParser')
# sys.path.append('/media/sda1/nana/opennre-pytorch')
from src.gen_mediate_para import hs_parse
# print(os.path.dirname(__file__))
data_dir=os.path.join('/media/sda1/nana/opennre-pytorch','mnre_data/176rels_data/need_data')
print(data_dir)
max_length=120
train_posseg=np.load(os.path.join(data_dir, 'train_posseg.npy'))

lstm_dict=dict()
ct=714000
mod=1000
lstm_parse_dir=os.path.join(data_dir,'f185_lstm_parse')
if not os.path.exists(lstm_parse_dir):
    os.mkdir(lstm_parse_dir)
for i in range(714000,len(train_posseg)):
    line = train_posseg[i]

    line = [tuple(i) for i in line]
    res = hs_parse(line)
    if len(res) < max_length:
        res = np.vstack((res, np.zeros((max_length - len(res), 100))))
    else:
        res = res[:max_length]
    lstm_dict[i]=res
    ct+=1
    if ct%mod==0:
        pc.dump(lstm_dict,open(os.path.join(lstm_parse_dir, 'train_{}.pc'.format(ct//mod)),mode='wb'))
        print('{} finished'.format(ct))
        lstm_dict=dict()
if len(lstm_dict)>0:
    pc.dump(lstm_dict, open(os.path.join(lstm_parse_dir, 'train_{}.pc'.format(ct // mod+1)), mode='wb'))

# lstm_dict=pc.load(open(os.path.join(data_dir,'lstm_parse', 'train_{}.pc'.format(2)),mode='rb'))
# for k,v in lstm_dict.items():
#     print(k,v)

# for k,v in lstm_dict:
#     print(k,v)