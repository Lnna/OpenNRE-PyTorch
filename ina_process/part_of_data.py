
import os
# 数量>500的关系抽取出来
remains={'P131','P22','P7','P17','P27','P150','P19','P47','P69','P20'}

def extract_data(ori_dir,new_dir,name='train'):
    ori_path=os.path.join(ori_dir,name+"_zh.txt")
    new_path=os.path.join(new_dir,name+"_zh.txt")
    fn=open(new_path,mode='w',encoding='utf-8')
    if name=='train':
        na=100000
    else:
        na=10000
    with open(ori_path,encoding='utf-8') as f:
        for line in f:
            if line.split()[4]=='NA' and na>=0:
                fn.write(line)
                na-=1
            elif line.split()[4] in remains:
                fn.write(line)


    fn.close()

def data_ct(dir,name='train'):
    new_path=os.path.join(dir,name+"_zh.txt")
    res=0
    with open(new_path,mode='r') as f:
        res=len(f.readlines())
    print('{} count: {}'.format(name,res))
if __name__=='__main__':
    ori_dir='/home/lnn/Documents/OpenNRE-Ina/OpenNRE-PyTorch/mnre_data'
    # new_dir='/home/lnn/Documents/OpenNRE-Ina/OpenNRE-PyTorch/mnre_data/thesis_data/raw_data'
    new_dir='/home/nana/Documents/pycharmforlinux/opennre-pytorch/mnre_data/thesis_data/raw_data'
    extract_data(ori_dir,new_dir,name='train')
    # data_ct(new_dir,name='valid')
    # data_ct(new_dir,name='test')
