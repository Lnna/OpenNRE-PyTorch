import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# 加载词典 pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('/media/sda1/nana/bert-rel/chinese_L-12_H-768_A-12')

# Tokenized input
text = "这是什么?不知道。"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['这', '是', '什','么',  '?', '不',  '知','道', '。']

# 将 token 转为 vocabulary 索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# 定义句子 A、B 索引
segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# 将 inputs 转为 PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)
segments_tensors = torch.tensor([segments_ids])

# 加载模型 pre-trained model (weights)
model = BertModel.from_pretrained('/media/sda1/nana/bert-rel/')
model.eval()

# GPU & put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

# 得到每一层的 hidden states
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors,output_all_encoded_layers=False)
# 模型 bert-base-uncased 有12层，所以 hidden states 也有12层
print(encoded_layers.size())
assert len(encoded_layers) == 1