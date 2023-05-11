import json
from tqdm import tqdm
with open('./datasets/train.jsonl', 'r', encoding='utf-8') as read_file, open('./FinRE.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        text = dict_['text']
        label_desc = dict_['label_desc']
        head = dict_['head']["mention"]
        tail = dict_['tail']["mention"]
        if label_desc != 'N/A':
            if text in slove_lines.keys():
                slove_lines[text].append(head + '-' + label_desc + '-' + tail)
            else:
                slove_lines[text] = []
                slove_lines[text].append(head + '-' + label_desc + '-' + tail)
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的实体间的关系、以及发生的事件。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('./datasets/valid.jsonl', 'r', encoding='utf-8') as read_file, open('./FinRE.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        text = dict_['text']
        label_desc = dict_['label_desc']
        head = dict_['head']["mention"]
        tail = dict_['tail']["mention"]
        if label_desc != 'N/A':
            if text in slove_lines.keys():
                slove_lines[text].append(head + '-' + label_desc + '-' + tail)
            else:
                slove_lines[text] = []
                slove_lines[text].append(head + '-' + label_desc + '-' + tail)
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的实体间的关系、以及发生的事件。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

import json
from tqdm import tqdm
with open('./FewCLUE/FewCLUE-main/datasets/csl/train_few_all.json', 'r', encoding='utf-8') as read_file, open('./FewCLUE_CSL.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        abst = dict_['abst']
        key_words = dict_['keyword']
        label = dict_['label']
        if label == '1' or label == 1:
            if abst in slove_lines.keys():
                slove_lines[abst].extend(key_words)
            else:
                slove_lines[abst] = key_words
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的关键词。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('./FewCLUE/FewCLUE-main/datasets/csl/dev_few_all.json', 'r', encoding='utf-8') as read_file, open('./FewCLUE_CSL.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        abst = dict_['abst']
        key_words = dict_['keyword']
        label = dict_['label']
        if label == '1' or label == 1:
            if abst in slove_lines.keys():
                slove_lines[abst].extend(key_words)
            else:
                slove_lines[abst] = key_words
    for key, value in tqdm(slove_lines.items()):
        item = {
            "instruction": "请帮我抽取出下面句子中的关键词。句子：”" + key + "”。",
            "input": "",
            "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('./FewCLUE/FewCLUE-main/datasets/csl/test_public.json', 'r', encoding='utf-8') as read_file, open('./FewCLUE_CSL.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        abst = dict_['abst']
        key_words = dict_['keyword']
        label = dict_['label']
        if label == '1' or label == 1:
            if abst in slove_lines.keys():
                slove_lines[abst].extend(key_words)
            else:
                slove_lines[abst] = key_words
    for key, value in tqdm(slove_lines.items()):
        item = {
            "instruction": "请帮我抽取出下面句子中的关键词。句子：”" + key + "”。",
            "input": "",
            "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')



import json
from tqdm import tqdm
with open('./FewCLUE/FewCLUE-main/datasets/tnews/train_few_all.json', 'r', encoding='utf-8') as read_file, open('./FewCLUE_TNEWS.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        sentence = dict_['sentence']
        key_words = dict_['keywords']
        if sentence in slove_lines.keys():
            slove_lines[sentence].extend(key_words.split(','))
        else:
            slove_lines[sentence] = key_words.split(',')
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的关键词。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

import json
from tqdm import tqdm
with open('./FewCLUE/FewCLUE-main/datasets/tnews/dev_few_all.json', 'r', encoding='utf-8') as read_file, open('./FewCLUE_TNEWS.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        sentence = dict_['sentence']
        key_words = dict_['keywords']
        if sentence in slove_lines.keys():
            slove_lines[sentence].extend(key_words.split(','))
        else:
            slove_lines[sentence] = key_words.split(',')
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的关键词。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

import json
from tqdm import tqdm
with open('./FewCLUE/FewCLUE-main/datasets/tnews/test_public.json', 'r', encoding='utf-8') as read_file, open('./FewCLUE_TNEWS.json', 'a', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    slove_lines = {}
    for line in read_lines:
        dict_ = json.loads(line)
        sentence = dict_['sentence']
        key_words = dict_['keywords']
        if sentence in slove_lines.keys():
            slove_lines[sentence].extend(key_words.split(','))
        else:
            slove_lines[sentence] = key_words.split(',')
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的关键词。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')


import json
from tqdm import tqdm
with open('./Doc2EDAG/Doc2EDAG-master/data/Data/train.json', 'r', encoding='utf-8') as read_file, open('./Doc2EDAG.json', 'a', encoding='utf-8') as write_file:
    list_ = json.load(read_file)
    slove_lines = {}
    for line in list_:
        dict_ = line[1]
        sentence = '\n'.join(dict_['sentences'])
        ann_valid_mspans = dict_['ann_valid_mspans']
        if sentence in slove_lines.keys():
            slove_lines[sentence].extend(ann_valid_mspans)
        else:
            slove_lines[sentence] = ann_valid_mspans
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的关键的实体。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

import json
from tqdm import tqdm
with open('./Doc2EDAG/Doc2EDAG-master/data/Data/dev.json', 'r', encoding='utf-8') as read_file, open('./Doc2EDAG.json', 'a', encoding='utf-8') as write_file:
    list_ = json.load(read_file)
    slove_lines = {}
    for line in list_:
        dict_ = line[1]
        sentence = '\n'.join(dict_['sentences'])
        ann_valid_mspans = dict_['ann_valid_mspans']
        if sentence in slove_lines.keys():
            slove_lines[sentence].extend(ann_valid_mspans)
        else:
            slove_lines[sentence] = ann_valid_mspans
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的关键的实体。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')

import json
from tqdm import tqdm
with open('./Doc2EDAG/Doc2EDAG-master/data/Data/test.json', 'r', encoding='utf-8') as read_file, open('./Doc2EDAG.json', 'a', encoding='utf-8') as write_file:
    list_ = json.load(read_file)
    slove_lines = {}
    for line in list_:
        dict_ = line[1]
        sentence = '\n'.join(dict_['sentences'])
        ann_valid_mspans = dict_['ann_valid_mspans']
        if sentence in slove_lines.keys():
            slove_lines[sentence].extend(ann_valid_mspans)
        else:
            slove_lines[sentence] = ann_valid_mspans
    for key, value in tqdm(slove_lines.items()):
        item = {
        "instruction": "请帮我抽取出下面句子中的关键的实体。句子：”" + key + "”。",
        "input": "",
        "output": "\n抽取出的内容如下：" + ','.join(value) + "。"
        }
        write_file.write(json.dumps(item, ensure_ascii=False) + '\n')
