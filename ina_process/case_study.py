import os
import numpy as np
import json
def find_rel(result_dir,weight_ori,weight_lstm):
    if not os.path.exists(result_dir):
        return
    ori_scores=np.load(os.path.join(result_dir, 'ori_predict_res.npy'))
    true_labels=np.load(os.path.join(result_dir, 'true_label_res.npy'))
    lstm_scores=np.load(os.path.join(result_dir, 'lstm_predict_res.npy'))
    test_scope=np.load(os.path.join(result_dir, 'data_test_scope.npy'))

    in_path = "../mnre_data/thesis_data/raw_data/json_data/test.json"
    ori_data=get_ori_data(in_path)
    result=[]#(sen_id,true_lstm_label,false_ori_label)
    for i in range(len(true_labels)):
        for id,(o,t,l) in enumerate(zip(ori_scores[i],true_labels[i],lstm_scores[i])):
            if id!=0 and t==1:
                if o<weight_ori and l>weight_lstm:
                    result.append((test_scope[i],id,np.argmax(ori_scores[i])))
    with open(os.path.join(result_dir,'case_result.txt'),mode='w') as f:
        for i in result:
            f.write(str(i)+'\n')
            for j in range(i[0][0],i[0][1]+1):
                f.write(str(ori_data[j])+'\n')

            f.write('\n\n')

def get_ori_data(file_name):
    print("Loading data file...")
    ori_data = json.load(open(file_name, "r"))
    print("Sorting data...")
    ori_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
    print("Finish sorting")
    return ori_data
if __name__=='__main__':
    result_dir='/home/nana/Documents/pycharmforlinux/opennre-pytorch/mnre_data/thesis_data/test_result'
    find_rel(result_dir,0.477,0.626)