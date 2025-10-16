
import os
import pandas as pd

isnull = lambda x : True if len(str(x).strip())==0 \
                or (str(x).lower() in ['nan', 'none', r'\N', r'\n'])   \
                else False
##################################################################################################################
DATA_DIR = r"data"

df_data = pd.read_excel(os.path.join(DATA_DIR, r"data.xlsx"))
df_data['input'] = df_data.apply(lambda x: str(x['sentence']), axis=1)
df_data['output'] = df_data['sentiment_annotation']
df_data['output'] = df_data['output'].fillna('')

df_data = df_data[df_data['output']!='']

df_data = df_data.sample(frac=1, random_state=10)
test_ratio = 0.1
df_test = df_data.iloc[:int(len(df_data.index)*test_ratio), :]
df_train = df_data.iloc[int(len(df_data.index)*test_ratio):, :]

MAX_LENGTH = 48
BATCH_SIZE = 16

print(df_data)


# Text processing

def zh2ch(s):

    import opencc
    converter = opencc.OpenCC('t2s')
    return converter.convert(s)

def remove_whitespace(text):

    import re
    text = re.sub(r"\s+", " ", text)
    return text

def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0020 <= inside_code and inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


df_train['input'] = df_train.apply(lambda x : remove_whitespace(x['input']), axis=1)
df_train['input'] = df_train.apply(lambda x : DBC2SBC(x['input']), axis=1)

df_test['input'] = df_test.apply(lambda x : remove_whitespace(x['input']), axis=1)
df_test['input'] = df_test.apply(lambda x : DBC2SBC(x['input']), axis=1)

import pickle
from sklearn.preprocessing import LabelEncoder
# 类别处理
label_encoder_file = os.path.join(DATA_DIR, 'label_encoder.pickle')
if os.path.exists(label_encoder_file):
    with open(label_encoder_file, 'rb') as fr:
        label_encoder = pickle.load(fr)
    label_train = label_encoder.transform(list(df_train['output']))
else:
    label_encoder = LabelEncoder()
    label_train = label_encoder.fit_transform(list(df_train['output']))
    with open(label_encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)

list_target_names = label_encoder.classes_
list_target = label_encoder.transform(label_encoder.classes_)
dict_label_map = {list_target_names[i]:list_target[i] for i in range(len(list_target_names))}
num_labels = len(list_target_names)


train_pairs = [[x, y] for x, y in zip(list(df_train['input']), label_train)]
test_pairs = [[x, y] for x, y in zip(list(df_test['input']), list(df_test['output']))]

dict_label_map


#### Data iterator
import random


class PairsLoader():
    def __init__(self, pairs, word2index, tokenizer, batch_size, max_length):
        self.word2index = word2index
        self.pairs = pairs
        self.batch_size = batch_size
        self.max_length = max_length
        self.position = 0
        self.tokenizer = tokenizer

    def load_single_pair(self):
        if self.position >= len(self.pairs):
            random.shuffle(self.pairs)
            self.position = 0
        single_pair = self.pairs[self.position]
        self.position += 1
        return single_pair

    def load(self):
        while True:
            input_batch = []
            output_batch = []
            mask_batch = []
            for i in range(self.batch_size):
                pair = self.load_single_pair()
                input_indexes, output_indexes, attn_masks = self.tokenizer(pair[0], pair[1], 'train')
                input_batch.append(input_indexes)
                output_batch.append(output_indexes)
                mask_batch.append(attn_masks)
            yield input_batch, mask_batch, output_batch


#  Model Definition

#  bert

import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = r"model_bert_classification"
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

init_model = r"bert-base-chinese"  # 中文

tokenizer_ori = BertTokenizer.from_pretrained(init_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(init_model, num_labels=num_labels)
model.resize_token_embeddings(len(tokenizer_ori))
model.to(device)


def tokenizer(x, y, type='train'):
    encodings_dict = tokenizer_ori(str(x), padding='max_length', truncation=True, max_length=MAX_LENGTH)
    input_indexes = encodings_dict['input_ids']
    attn_masks = encodings_dict['attention_mask']
    output_indexes = y
    return input_indexes, output_indexes, attn_masks


import torch
from tqdm import tqdm
from torch.autograd import Variable
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

EPOCH = 10
LR = 1e-5
WARMUP_STEPS = 1e2
EPSILON = 1e-8
# 初始化优化器
optimizer = AdamW(model.parameters(), lr=LR, eps=EPSILON)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                            num_training_steps=len(train_pairs) * EPOCH)

train_dataloader = PairsLoader(train_pairs, None, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

plot_losses = []
for epoch in range(EPOCH):
    print_loss_total = 0
    train_step = int(len(train_pairs) / BATCH_SIZE)

    with tqdm(total=train_step, desc=f'Train Epoch {epoch}/{EPOCH}', postfix=dict) as pbar:
        for i in range(train_step):
            input_index, mask_batch, output_index = next(train_dataloader.load())
            input_variable = Variable(torch.tensor(input_index)).to(device)
            output_variable = Variable(torch.LongTensor(output_index)).to(device)
            mask_variable = Variable(torch.tensor(mask_batch)).to(device)

            model.zero_grad()
            outputs = model(input_variable, attention_mask=mask_variable, labels=output_variable)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            print_loss_total += loss

            pbar.set_postfix(**{'Train Loss': float(print_loss_total / (i + 1)), 'every loss': float(loss)})
            pbar.update(1)
    plot_losses.append(float(print_loss_total / (i + 1)))

    output_dir = os.path.join(MODEL_DIR, str(epoch))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer_ori.save_pretrained(output_dir)


df_plot = pd.DataFrame([[i, x] for i, x in enumerate(plot_losses)], columns=['epoch', 'loss'])
fig = df_plot.plot(x='epoch', y='loss', title='Training Loss')
fig = fig.get_figure()
fig.savefig(os.path.join(MODEL_DIR, "train_loss.png"))

# test


import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn import metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer_ori = BertTokenizer.from_pretrained(os.path.join("model_bert_classification", '0'))


def evaluate(prompt, model):
    prompt = tokenizer_ori.encode(str(prompt), truncation=True, max_length=MAX_LENGTH)
    prompt = torch.tensor(prompt).unsqueeze(0).to(device)
    output = model(prompt)
    y_pred_label = output['logits'].argmax(dim=1)
    pred_name = label_encoder.inverse_transform([y_pred_label.item()])[0]

    return pred_name


for i in range(10):
    epoch = str(i)
    model = BertForSequenceClassification.from_pretrained(os.path.join("model_bert_classification", epoch))

    model.to(device)
    model.eval()

    list_pred = []
    for pair in test_pairs:
        pred = evaluate(pair[0], model)
        list_pred.append(pred)

    list_label = list(df_test['output'])

    print(epoch)

    score = metrics.accuracy_score(list_label, list_pred)
    print(init_model + "accuracy:   %0.3f" % score)

    report = metrics.classification_report(
        list_label, list_pred)
    print(init_model + "classification report:")
    print(report)
    print('_' * 80)

# %%

epoch = 7
init_model = os.path.join("model_bert_classification", str(epoch))
tokenizer_ori = BertTokenizer.from_pretrained(init_model)
model = BertForSequenceClassification.from_pretrained(init_model)

model.to(device)
model.eval()

list_result = []
for pair in test_pairs:
    pred = evaluate(pair[0], model)
    list_result.append([pair[0], pair[1], pred])
df = pd.DataFrame(list_result, columns=['text', 'label', 'pred'])
df.to_excel(os.path.join(init_model, "测试集预测结果.xlsx"), index=False)

score = metrics.accuracy_score(list(df['label']), list(df['pred']))
print(init_model + "_accuracy:   %0.3f" % score)
print('_' * 80)

list_label = label_encoder.transform(list(df['label']))
list_pred = label_encoder.transform(list(df['pred']))

report = metrics.classification_report(
    list_label, list_pred, target_names=[str(x) for x in label_encoder.classes_],
    labels=label_encoder.transform(label_encoder.classes_))
print(init_model + "classification report:")
print(report)

import torch.nn.functional as F


def evaluate1(prompt, model):
    prompt = tokenizer_ori.encode(str(prompt), truncation=True, max_length=MAX_LENGTH)
    prompt = torch.tensor(prompt).unsqueeze(0).to(device)
    output = model(prompt)
    y_pred_label = output['logits'].argmax(dim=1)
    pred_name = label_encoder.inverse_transform([y_pred_label.item()])[0]
    p = F.softmax(output['logits'], dim=-1)

    if pred_name == 3:
        score = p[0][dict_label_map[pred_name]].item()
    elif pred_name == 1:
        score = -1 * p[0][dict_label_map[pred_name]].item()
    else:
        score = 0

    return pred_name, score


import os
import pandas as pd

df = pd.read_csv(os.path.join("data", "全部数据.csv"))

list_res1 = []
list_res2 = []
for s in list(df['sentence']):
    temp = evaluate1(s, model)
    list_res1.append(temp[0])
    list_res2.append(temp[1])

df['sentiment_polarity'] = list_res1
df['sentiment_score'] = list_res2

df.to_csv(os.path.join('data', 'results.csv'), index=False)
df


