操作：

1) screen -r gen_data上现在是cons+关系抽取+176个关系的数据的实验。
2) screen -r f187上是f1=87的mparser的生成

1. 11个关系到176个关系时模型的改动：

config：num_classes=176,max_length=120

报错：CUDNN_STATUS_NOT_INITIALIZED CUDNN_STATUS_INTERNAL_ERROR

原因分析：改成在cpu上运行，错误为

RuntimeError: index out of range at /opt/conda/condabld/pytorch_1532576128691/work/aten/src/TH/generic/THTensorMath.cpp:352

最终则是一些参数设置错误

如果cpu上没有问题则应该是cuda的问题了，此时可能是cache的问题，

rm -rf ~/.nv

2. 数据的更新生成： 
gen_data.py ： 生成关系抽取的train test数据，包括分词和词性标注后的文本，用于后续parser生成。

mparser.py ： 生成关系抽取文本的cons，用于关系抽取中。

3. 不同f1 parser的生成：

