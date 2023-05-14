import torch
import numpy as np
import os.path as osp
import json
import pkuseg
import re
import pandas as pd
from decimal import Decimal
import random
from bs4 import BeautifulSoup
from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig


def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, modify_dataset = False,is_training = True,max_seq_length = 512,hidden_size = 768,original_train_path = "data/train.json",original_test_path = "data/test_B.json",train_path = "data/train_V3.txt",test_path = "data/test_B_V3.txt",prompt = False,input_format = "statistics+320+64",model_name = "nezha-base-cn"):
        self.root = root
        if 'train' in root:
            self.input_file = './data/train.json'
        elif 'eval' in root:
            self.input_file = './data/test_B.json'
        self.length = 0
        self.is_train = is_training
        self.prompt = prompt
        self.max_seq_length = max_seq_length
        self.input_format = input_format
        if modify_dataset == True:
            if is_training:
                self.data_to_csv(original_train_path, train_path)
            else:
                self.data_to_csv(original_test_path, test_path)
        if is_training:
            self.df = self.read_data(train_path)
        else:
            self.df = self.read_data(test_path)
        if model_name == "xlnet-base-cn" :
            self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
        elif "nezha" in model_name:
            self.config = BertConfig("NEZHA/bert_config.json")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese",config = self.config)
        else:
            raise NameError("未找到该模型，请重新输入或配置该模型！")
        super().__init__(root, transform, pre_transform)

    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['./data/train.json', './data/test_B.json']

    # 返回process方法所需的保存文件名。之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in range(self.length)]

    def read_data(self,path):
        df = pd.read_csv(path)
        return df

    def data_to_csv(self, origin, destination):
        data = []
        if 'train' in origin:
            summary_path = './data/train_summary.json'
        elif 'test' in origin:
            summary_path = './data/eval_summary.json'
        else:
            assert False
        with open(origin, mode='r', encoding='utf-8') as f, open(summary_path, 'r', encoding='utf-8') as summary_file:
            summary_lines = summary_file.readlines()
            for f_index, line in enumerate(f.readlines()):
                json_info = json.loads(line)
                summary_content = json.loads(summary_lines[f_index])['content']
                url = json_info['url']
                entity_length = 0
                if "entities" in json_info:
                    entity_length = len(json_info['entities'])  # 实体的个数
                title_length = len(json_info['title'])
                content_length = len(json_info['content'])
                cnt_1 = json_info['content'].count("，")  # 内容中的逗号数
                cnt_2 = json_info['content'].count("：")  # 内容中的冒号数
                cnt_3 = json_info['content'].count("。")  # 内容中的句号数

                cnt_4 = 0
                cnt_5 = 0
                if self.is_train:
                    y = int(json_info['label'])
                    if self.prompt:
                        y = 3221 if y == 1 else 1415
                if self.input_format == "statistics+320+64":
                    x = "[CLS]" + json_info["title"] + "[SEP]" + summary_content[:320] + "[SEP]" + summary_content[
                                                                                                        -64:] + "[SEP]"
                elif self.input_format == "statistics+512+128":
                    x = "[CLS]" + json_info["title"] + "[SEP]" + summary_content[:512] + "[SEP]" + summary_content[
                                                                                                        -128:] + "[SEP]"
                else:
                    x = "[CLS]" + json_info["title"] + "[SEP]" + summary_content[:200] + "[SEP]"
                if self.prompt:
                    x = "[CLS]" + "是否为优质文章：[MASK]。[SEP]" + json_info["title"] + "[SEP]" + summary_content[
                                                                                                 :320] + "[SEP]"
                if "entities" in json_info:
                    for entity in json_info['entities']:
                        if "entities" in self.input_format:
                            x += entity + "[SEP]"
                        cnt_4 += len(json_info['entities'][entity]["entity_baike_info"])
                        cnt_5 += len(json_info['entities'][entity]['co-occurrence'])

                if self.is_train:

                    data.append(
                        [x[:self.max_seq_length], entity_length, title_length, content_length, cnt_1, cnt_2, cnt_3,
                         cnt_4, cnt_5, y])
                    # if y == 1 and random.random() < 0.05:
                    #     data.append([x[:self.max_seq_length], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,y])
                else:

                    data.append(
                        [url, x[:self.max_seq_length], entity_length, title_length, content_length, cnt_1, cnt_2, cnt_3,
                         cnt_4, cnt_5])
                    # data.append([url,x_2[:512], entity_length,title_length,content_length,cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,cnt_6,cnt_7])

            f.close()
        df = pd.DataFrame(data)
        # df.columns = ['x','entity_length','title_length','content_length','cnt_1','cnt_2','cnt_3','cnt_4','cnt_5','cnt_6''cnt_7','y']
        df.to_csv(destination, index=None)

    def convert_examples_to_features(self, x_str, statistics):
        lis = self.tokenizer.tokenize(x_str)
        if self.prompt:
            masked_index = lis.index("[MASK]")
        else:
            masked_index = 0
        input_ids = self.tokenizer.convert_tokens_to_ids(lis)
        if "statistics" in self.input_format:  # 以取对数的形式加入统计信息，加1是为了防止log函数入参为0导致运算错误。
            for item in statistics:
                input_ids.insert(1, int(np.log(item + 1)))

        if len(input_ids) > self.max_seq_length:  # 对大于max_seq_len的数据进行处理。
            input_ids = input_ids[:self.max_seq_length - 1]
            input_ids.append(102)
        masked_attention = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        masked_attention += ([0] * padding_length)
        input_ids = torch.LongTensor(input_ids)
        masked_attention = torch.LongTensor(masked_attention)
        return input_ids, masked_attention, masked_index

    # 生成数据集所用的方法
    def process(self):
        seg = pkuseg.pkuseg()
        total_length = 0
        if 'train' in self.input_file:
            self.summary_file = './data/train_summary.json'
            self.logits_difference_file = './data/train_logits_difference_1024.json'
        else:
            self.summary_file = './data/eval_summary.json'
            self.logits_difference_file = './data/eval_logits_difference_1024.json'
        with open(self.input_file, 'r', encoding='utf-8') as read_file, \
                open(self.summary_file, 'r', encoding='utf-8') as summary_file, \
                open(self.logits_difference_file, 'r', encoding='utf-8') as difference_file:
            summary_lines = summary_file.readlines()
            difference_lines = difference_file.readlines()
            for idx, data_item in enumerate(tqdm(read_file.readlines(), desc='reading corpus')):
                data_item = json.loads(data_item)
                summary_item = json.loads(summary_lines[idx])
                difference_item = json.loads(difference_lines[idx])
                assert data_item['url'] == summary_item['url'] == difference_item['url']
                if data_item is not None:
                    total_length += 1
                    # 原始信息读取
                    url = data_item['url']
                    title = data_item['title']
                    pub_time = data_item['pub_time']
                    content = summary_item['content']
                    predict_text = difference_item['predict_text']
                    input_text = difference_item['input_text']


                    entities = data_item['entities']
                    label = data_item['label'] if "label" in data_item.keys() else "0"
                    bf_title = BeautifulSoup(title, 'html.parser')
                    bf_content = BeautifulSoup(content, 'html.parser')
                    title = bf_title.get_text().replace("{", "").replace("}", "").replace("$", "")
                    title = title if title is not None else "None_title"
                    content = bf_content.get_text().replace("{", "").replace("}", "").replace("$", "")
                    content = content if content is not None else "None_content"

                    # 处理实体信息
                    key_list = [key for key in entities.keys()]
                    # 去除重复的共现边（key，value）,(value, key)属于重复
                    co_occurrence = []
                    co_occurrence = [(key, value) for key in key_list for value in entities[key]["co-occurrence"] if
                                     (value, key) not in co_occurrence]

                    key_tensor = [[0 for i in range(3)] for key in key_list]

                    key_id = {key: index for index, key in enumerate(key_list)}
                    entities_co_occurrence = [[key_id[key], key_id[value]] for key, value in co_occurrence]
                    entities_co_occurrence.extend([[key_id[value], key_id[key]] for key, value in co_occurrence])
                    key_has_message = []
                    message_belong_key = []
                    message_tensor = []
                    count = 0
                    message_text = []

                    for key in key_list:

                        # 对每个实体的信息进行单独处理
                        message_text.extend(
                            [item["name"] + ":" + "".join(item["value"]) for item in entities[key]["entity_baike_info"]])

                        message_tensor.extend([[0 for i in range(3)] for item in entities[key]["entity_baike_info"]])

                        for item in entities[key]["entity_baike_info"]:
                            key_has_message.append([key_id[key], count])
                            message_belong_key.append([count, key_id[key]])
                            count += 1

                    # 将相关张量转为图节点
                    data = HeteroData()
                    x_father = torch.tensor([[0 for i in range(3)]], dtype=torch.float)
                    data['father'].x = x_father
                    entities_key_tensor = torch.tensor(np.array(key_tensor), dtype=torch.float)
                    data["entities"].x = entities_key_tensor
                    entities_message_tensor = torch.tensor(np.array(message_tensor), dtype=torch.float)
                    data["message"].x = entities_message_tensor
                    pub_time_tensor = torch.tensor([[0 for i in range(3)]], dtype=torch.float)
                    # data['pub_time'].x = pub_time_tensor
                    # url_tensor = torch.tensor([[0 for i in range(3)]], dtype=torch.float)
                    # data['url'].x = url_tensor
                    # title_tensor = torch.tensor([[0 for i in range(3)]], dtype=torch.float)
                    # data['title'].x = title_tensor

                    # 链接相关节点的边

                    # 双向边
                    key_has_message = torch.tensor(key_has_message, dtype=torch.long)
                    key_has_message = key_has_message.t().contiguous()
                    data["entities", "has", "message"].edge_index = key_has_message
                    message_belong_key = torch.tensor(message_belong_key, dtype=torch.long)
                    message_belong_key = message_belong_key.t().contiguous()
                    data["message", "belong", "entities"].edge_index = message_belong_key
                    entities_co_occurrence = torch.tensor(entities_co_occurrence, dtype=torch.long)
                    entities_co_occurrence = entities_co_occurrence.t().contiguous()
                    data["entities", "co_occurrence", "entities"].edge_index = entities_co_occurrence


                    # 单向边
                    entities_tensor_length = len(entities_key_tensor)
                    entities_refer_father = torch.tensor([[i for i in range(entities_tensor_length)],
                                                          [0 for i in range(entities_tensor_length)]],
                                                         dtype=torch.long)
                    data["entities", "refer", "father"].edge_index = entities_refer_father
                    # title_refer_father = torch.tensor([[0], [0]], dtype=torch.long)
                    # data["title", "refer", "father"].edge_index = title_refer_father
                    # url_refer_father = torch.tensor([[0], [0]], dtype=torch.long)
                    # data["url", "refer", "father"].edge_index = url_refer_father
                    # pub_time_refer_father = torch.tensor([[0], [0]], dtype=torch.long)
                    # data["pub_time", "refer", "father"].edge_index = pub_time_refer_father

                    data.y = torch.tensor([int(label)])
                    try:
                        content_origin = data_item['content']
                        content_zero_number = 0
                        while len(content_origin) >= 20:
                            content_zero_number += 1
                            content_origin = content_origin[1024:]

                        predict_text_str = predict_text.strip('[').strip(']').split(', ')
                        predict_text_new = [Decimal(a).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP") for a in predict_text_str]
                        predict_text = [float(a) for a in predict_text_new]

                        input_text_str = input_text.strip('[').strip(']').split(', ')
                        input_text_new = [Decimal(a).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP") for a in input_text_str]
                        input_text = [float(a) for a in input_text_new]

                        input_zero_number = 0
                        input_zero_position = []
                        for index, item in enumerate(input_text_str):
                            if item == '0.0':
                                input_zero_number += 1
                                input_zero_position.append(index)
                        assert content_zero_number == input_zero_number and input_zero_position[-1] == (len(predict_text)-1)
                    except Exception as e:
                        print(content_zero_number)
                        print(input_zero_number)
                        print(len(content_origin))
                        print(input_text_str)
                        assert False

                    predict_text_result = []
                    if len(input_zero_position) > 2:
                        for index in range(len(input_zero_position)):
                            if input_zero_position[index] < len(predict_text)-1:
                                if index != 0:
                                    predict_text_result += predict_text[(input_zero_position[index - 1] + 1):(input_zero_position[index])]
                                else:
                                    predict_text_result += predict_text[:input_zero_position[0]]
                            elif input_zero_position[index] == len(predict_text)-1:
                                predict_text_result += predict_text[(input_zero_position[index-1]+1):-1]
                            else:
                                assert False
                    else:
                        if len(input_zero_position) == 2:
                            predict_text_result += predict_text[:input_zero_position[0]]
                            predict_text_result += predict_text[(input_zero_position[0]+1):-1]
                        elif len(input_zero_position) == 1:
                            predict_text_result += predict_text[:-1]
                        else:
                            assert False

                    assert len(predict_text_result) == (len(predict_text) - len(input_zero_position))

                    input_text_result = []
                    if len(input_zero_position) > 2:
                        for index in range(len(input_zero_position)):
                            if input_zero_position[index] < len(input_text) - 1:
                                if index != 0:
                                    input_text_result += input_text[
                                                         (input_zero_position[index - 1] + 1):(
                                                         input_zero_position[index])]
                                else:
                                    input_text_result += input_text[:input_zero_position[0]]
                            elif input_zero_position[index] == len(input_text) - 1:
                                input_text_result += input_text[(input_zero_position[index - 1] + 1):-1]
                            else:
                                assert False
                    else:
                        if len(input_zero_position) == 2:
                            input_text_result += input_text[:input_zero_position[0]]
                            input_text_result += input_text[(input_zero_position[0] + 1):-1]
                        elif len(input_zero_position) == 1:
                            input_text_result += input_text[:-1]
                        else:
                            assert False

                    assert len(input_text_result) == (len(input_text) - len(input_zero_position))


                    assert len(input_text_result) == len(predict_text_result)
                    if len(predict_text_result) >= 2048:
                        predict_text = predict_text_result[:2048]
                        input_text = input_text_result[:2048]
                    else:
                        division = int(2048 / len(predict_text_result)) + 1
                        predict_text = []
                        input_text = []
                        for i in range(division):
                            predict_text += predict_text_result
                            input_text += input_text_result
                        predict_text = predict_text[:2048]
                        input_text = input_text[:2048]
                    assert len(input_text) == len(predict_text) == 2048

                    # todo 验证input_text和predict_text的正确性

                    logits_difference_result = [a - b for a, b in zip(predict_text, input_text)]

                    logits_difference_input = [predict_text, input_text, logits_difference_result]
                    # logits_difference_input = torch.tensor(logits_difference_input, dtype=torch.float)


                    origin_tuple = None
                    x_str = self.df.iloc[idx, 0]
                    statistics = self.df.iloc[idx, 1:11]
                    if not self.is_train:
                        x_str = self.df.iloc[idx, 1]
                        statistics = self.df.iloc[idx, 2:12]
                    input_ids, masked_attention, masked_index = self.convert_examples_to_features(x_str, statistics)
                    if self.is_train:
                        if self.prompt:
                            origin_tuple = [[input_ids, masked_attention, masked_index], [
                                torch.Tensor(self.df.iloc[idx, 1:9])], torch.LongTensor([self.df.iloc[idx, -1]])]
                        else:
                            origin_tuple = [[input_ids, masked_attention], [torch.Tensor(self.df.iloc[idx, 1:9])], torch.LongTensor(
                                [self.df.iloc[idx, -1]])]
                    else:
                        origin_tuple = [[self.df.iloc[idx, 0]], [input_ids, masked_attention], [
                            torch.Tensor(self.df.iloc[idx, 2:10])]]

                    data.dict_ = {'origin_tuple': origin_tuple, 'logits_difference_input': logits_difference_input, 'entities': key_list, 'message': message_text, "pub_time": pub_time, "title": title, "url": url, "content": content}

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            self.length = total_length

    def len(self):
        return self.length

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


if __name__ == "__main__":
    pass
