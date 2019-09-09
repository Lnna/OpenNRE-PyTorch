from os import path
import json

def relid2json(rel_id_path,json_path):

    rel2id={}
    f=open(rel_id_path)
    for line in f.readlines():
        pair=line.strip('\n').split('\t')
        rel2id[pair[0]]=int(pair[1])
    f.close()

    json.dump(rel2id,open(json_path,mode='w'),ensure_ascii=False)


def samples2json(sample_path,json_path):

    res=[]
    f=open(sample_path)
    for line in f.readlines():
        per=line.strip('\n').strip('###END###').strip().split('\t')
        d={'sentence':per[5],'head':{'id':per[0],'word':per[2]},'tail':{'id':per[1],'word':per[3]},'relation':per[4]}
        res.append(d)
    f.close()
    json.dump(res,open(json_path,mode='w'),ensure_ascii=False)

def wordvec2json(wordvec_path, new_path,json_path):
    if not path.exists(new_path):

        from gensim.models.keyedvectors import KeyedVectors

        model = KeyedVectors.load_word2vec_format(wordvec_path,unicode_errors='ignore', binary=True)
        model.save_word2vec_format(new_path, binary=False)
    res=[]
    with open(new_path,mode='r') as f:
        for line in f:
            per = line.strip('\n').split()
            if len(per)!=51:
                continue
            try:
                d = {'word': per[0], 'vec': [float(i) for i in per[1:51]]}
            except ValueError:
                continue
            res.append(d)

    json.dump(res,open(json_path,mode='w'),ensure_ascii=False)



mnre_dir=path.join(path.dirname(__file__),'mnre_data/new_data')

# rel2id=relid2json(path.join(mnre_dir,'relation2id.txt'),path.join(mnre_dir, 'json_data','rel2id.json'))

# samples2json(path.join(mnre_dir,'train_zh.txt'),path.join(mnre_dir, 'json_data', 'train.json'))
# samples2json(path.join(mnre_dir,'try.txt'),path.join(mnre_dir, 'new_data', 'valid.json'))
# samples2json(path.join(mnre_dir,'valid_zh.txt'),path.join(mnre_dir, 'json_data', 'valid.json'))
# samples2json(path.join(mnre_dir,'test_zh.txt'),path.join(mnre_dir, 'json_data', 'test.json'))

wordvec2json(path.join(mnre_dir,'vec_zh.bin'), path.join(mnre_dir,  'word_vec.txt'),path.join(mnre_dir, 'word_vec.json'))
