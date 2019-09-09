
def group_by_relation(path):
    relations=dict()
    with open(path,encoding='utf-8') as f:
        for line in f:
            if relations.get(line.split()[4],-1)==-1:
                relations[line.split()[4]]=1
            else:
                relations[line.split()[4]]=relations[line.split()[4]]+1
    relations=sorted(relations.items(),key=lambda a:a[1],reverse=True)
    return relations
    # print("relation_id count")
    # for r in relations:
    #     print(r[0]+" "+str(r[1]))

def group_by_ner(path):
    ners=dict()
    ct=0
    with open(path,encoding='utf-8') as f:
        for line in f:
            ct+=1
            if line.split()[4]=='NA':
                continue
            if ners.get(' '.join(line.split()[2:4]),-1)==-1:
                ners[' '.join(line.split()[2:4])]=1
            else:
                ners[' '.join(line.split()[2:4])] =ners[' '.join(line.split()[2:4])]+1
    print(len(ners))
    print(ct)

import itertools
import copy
import matplotlib.pyplot as plt
def cdf(data):
    x_axis=[10,50,100,500,1000,5000,10000,100000,1000000]
    y_axis=copy.deepcopy([0]*9)

    for r,ct in data:
        for i,x in enumerate(x_axis):
            if ct<=x:
                y_axis[i]+=1
                break


    fig = plt.figure(1)

    ax = fig.add_subplot(111)

    ax.plot([i for i in range(0,9)], y_axis)

    ax.set_title('relation-sentences distribution', bbox={'facecolor': '0.8', 'pad': 5})

    ax.set_xticks([i for i in range(0,9)])
    ax.set_xticklabels([str(i) for i in x_axis],rotation='vertical')

    plt.grid(True)
    plt.legend()
    plt.show()



if __name__=="__main__":
    data_path="/home/nana/Documents/pycharmforlinux/opennre-pytorch/mnre_data/thesis_data/raw_data/test_zh.txt"
    # data_path="/home/lnn/Documents/OpenNRE-Ina/OpenNRE-PyTorch/mnre_data/valid_zh.txt"
    # cdf(group_by_relation(data_path))
    group_by_ner(data_path)
    # a={"a":3,"b":1,"c":2}
    # print(sorted(a.items(),key=lambda a:a[1],reverse=True))