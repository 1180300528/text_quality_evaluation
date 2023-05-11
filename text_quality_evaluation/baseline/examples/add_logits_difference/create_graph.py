import torch
import numpy as np
import os.path as osp
import json
import pkuseg
import re
from decimal import Decimal
import random
from bs4 import BeautifulSoup
from torch_geometric.data import Dataset
from torch_geometric.data import Data, HeteroData
from tqdm import tqdm
from transformers import AutoTokenizer


def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root = root
        if 'train' in root:
            self.input_file = '../../../data/new/train.json'
        elif 'eval' in root:
            self.input_file = '../../../data/new/eval.json'
        self.length = 0
        super().__init__(root, transform, pre_transform)

    # 返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['../../data/new/train.json', '../../data/new/eval.json']

    # 返回process方法所需的保存文件名。之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in range(self.length)]

    # 生成数据集所用的方法
    def process(self):
        seg = pkuseg.pkuseg()
        total_length = 0

        # with open(self.input_file, 'r', encoding='utf-8') as read_file:
        #     for idx, data_item in enumerate(tqdm(read_file.readlines(), desc='reading corpus')):
        #         data_item = json.loads(data_item)
        #         if data_item is not None:
        #             total_length += 1
        #             # 原始信息读取
        #             url = data_item['url']
        #             title = data_item['title']
        #             pub_time = data_item['pub_time']
        #             content = data_item['content']
        if 'train' in self.input_file:
            self.summary_file = '../../../data/new/train_summary.json'
            self.logits_difference_file = '../../../data/new/train_logits_difference_1024.json'
        else:
            self.summary_file = '../../../data/new/eval_summary.json'
            self.logits_difference_file = '../../../data/new/eval_logits_difference_1024.json'
        with open(self.input_file, 'r', encoding='utf-8') as read_file, \
                open(self.summary_file, 'r', encoding='utf-8') as summary_file, \
                open(self.logits_difference_file, 'r', encoding='utf-8') as difference_file:
            summary_lines = summary_file.readlines()
            difference_lines = difference_file.readlines()
            for idx, data_item in enumerate(tqdm(read_file.readlines(), desc='reading corpus')):
                data_item = json.loads(data_item)
                summary_item = json.loads(summary_lines[idx])
                difference_item = json.load(difference_lines[idx])
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

                    # # 处理文本信息，根据实体进行切割然后分词，保证实体不会因为分词而被切割
                    # text = title + content
                    # for key in key_list:
                    #     text = text.replace(str(key), "entities_split_sigin" + str(key) + "entities_split_sigin")
                    # text_list = re.split(r'entities_split_sigin', text)
                    # text_node_list = []
                    # text_tensor = []
                    # entities_in_text = []
                    # for item in text_list:
                    #     if item not in key_list:
                    #         # for nosize in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', '-', '.', ':', '(', ')', '<', '>', '/']:
                    #         #     item = item.replace(nosize, '@')
                    #         item = re.sub('[a-zA-Z1-90\.\\\-\_\:\/\(\)\<\>]', '', item)
                    #         for one in seg.cut(item):
                    #             list__ = [two for two in one.split('@') if two != '']
                    #             text_node_list.extend(list__) if len(list__) != 0 else text_node_list
                    #     else:
                    #         text_node_list.extend([item])
                    #
                    # # text_tensor = self._get_need_tensor(text_node_list, block_size=128, pre_text_len=-8)
                    # text_tensor = [[0 for i in range(5)] for j in text_node_list]
                    #
                    # for index, item in enumerate(text_node_list):
                    #
                    #     # # 调用模型，获得tensor张量结果
                    #     # tokenized_result = tokenizer(item, return_offsets_mapping=True, return_tensors='pt', padding=True, truncation=True)
                    #     # with torch.no_grad():
                    #     #     ids, masks = tokenized_result['input_ids'], tokenized_result['attention_mask'].to(
                    #     #         DEVICE)
                    #     #     result_repr = roberta(ids, masks)
                    #     #     result_repr = result_repr.detach().to('cpu').numpy()
                    #     #     text_tensor.append(result_repr)
                    #
                    #     if item in key_list:
                    #         entities_in_text.extend([[key_id[item], index]])
                    #         # todo 是否创建双向边， 还是选择只采用单向边
                    #         # text_has_entities.extend([[index, key_id[item]]])
                    # # 将相关张量转为图节点
                    # data = HeteroData()
                    # x_father_1 = [[0 for i in range(100)]]
                    # x_father_2 = [[0 for i in range(100)]]
                    # x_father_3 = [[0 for i in range(100)]]
                    # x_father_1 = torch.tensor(x_father_1, dtype=torch.float)
                    # x_father_2 = torch.tensor(x_father_2, dtype=torch.float)
                    # x_father_3 = torch.tensor(x_father_3, dtype=torch.float)
                    # data['father_last'].x = x_father_1
                    # data["text_father_node"].x = x_father_2
                    # data["entities_father_node"].x = x_father_3
                    # text_tensor = torch.tensor(np.array(text_tensor), dtype=torch.float)
                    # data["text_node"].x = text_tensor
                    # entities_key_tensor = torch.tensor(np.array(key_tensor), dtype=torch.float)
                    # data["entities"].x = entities_key_tensor
                    # entities_message_tensor = torch.tensor(np.array(message_tensor), dtype=torch.float)
                    # data["message"].x = entities_message_tensor
                    #
                    # # 链接相关节点的边
                    # key_has_message = torch.tensor(key_has_message, dtype=torch.long)
                    # key_has_message = key_has_message.t().contiguous()
                    # data["entities", "has", "message"].edge_index = key_has_message
                    # message_belong_key = torch.tensor(message_belong_key, dtype=torch.long)
                    # message_belong_key = message_belong_key.t().contiguous()
                    # data["message", "belong", "entities"].edge_index = message_belong_key
                    # entities_co_occurrence = torch.tensor(entities_co_occurrence, dtype=torch.long)
                    # entities_co_occurrence = entities_co_occurrence.t().contiguous()
                    # data["entities", "co_occurrence", "entities"].edge_index = entities_co_occurrence
                    # entities_in_text = torch.tensor(entities_in_text, dtype=torch.long)
                    # entities_in_text = entities_in_text.t().contiguous()
                    # data["entities", "in", "text_node"].edge_index = entities_in_text
                    #
                    # # todo 需确保这一部分的正确性
                    # # text_tensor_length = len(text_node_list)
                    # # entities_tensor_length = len(key_list)
                    # text_tensor_length = len(text_tensor)
                    # entities_tensor_length = len(entities_key_tensor)
                    #
                    # text_next2_text = torch.tensor([[i for i in range(0, text_tensor_length - 1)],
                    #                                 [i for i in range(1, text_tensor_length)]], dtype=torch.long)
                    # data["text_node", "next_to", "text_node"].edge_index = text_next2_text
                    # text_refer_father = torch.tensor([[i for i in range(text_tensor_length)],
                    #                                   [0 for i in range(text_tensor_length)]], dtype=torch.long)
                    # data["text_node", "refer", "text_father_node"].edge_index = text_refer_father
                    # entities_refer_father = torch.tensor([[i for i in range(entities_tensor_length)],
                    #                                       [0 for i in range(entities_tensor_length)]],
                    #                                      dtype=torch.long)
                    # data["entities", "refer", "entities_father_node"].edge_index = entities_refer_father
                    # last_refer_father = torch.tensor([[0], [0]], dtype=torch.long)
                    # data["entities_father_node", "refer", "father_last"].edge_index = last_refer_father
                    # # 当进行额外的数据扩充时，需修改last_refer_father为不同变量以进行区分
                    # # last_refer_father = torch.tensor([[0], [0]], dtype=torch.long)
                    # data["text_father_node", "refer", "father_last"].edge_index = last_refer_father

                    # 加入标签



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

                    content_origin = data_item['content']
                    content_zero_number = 0
                    while len(content_origin) >= 20:
                        content_zero_number += 1
                        content_origin = content_origin[1024:]

                    predict_text = predict_text.strip('[').strip(']').split(', ')
                    predict_text_new = [Decimal(a).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP") for a in predict_text]
                    predict_text = [float(a) for a in predict_text_new]

                    predict_zero_number = 0
                    predict_zero_position = []
                    for index, item in enumerate(predict_text):
                        if abs(item) < 1e-6:
                            predict_zero_number += 1
                            predict_zero_position.append(index)
                    assert content_zero_number == predict_zero_number and predict_zero_position[-1] == (len(predict_text)-1)

                    predict_text_result = []
                    if len(predict_zero_position) > 2:
                        for index in range(len(predict_zero_position)):
                            if predict_zero_position[index] < len(predict_text)-1:
                                if index != 0:
                                    predict_text_result += predict_text[(predict_zero_position[index - 1] + 1):(predict_zero_position[index])]
                                else:
                                    predict_text_result += predict_text[:predict_zero_position[0]]
                            elif predict_zero_position[index] == len(predict_text)-1:
                                predict_text_result += predict_text[(predict_zero_position[index-1]+1):-1]
                            else:
                                assert False
                    else:
                        if len(predict_zero_position) == 2:
                            predict_text_result += predict_text[:predict_zero_position[0]]
                            predict_text_result += predict_text[(predict_zero_position[0]+1):-1]
                        elif len(predict_zero_position) == 1:
                            predict_text_result += predict_text[:-1]
                        else:
                            assert False

                    assert len(predict_text_result) == (len(predict_text) - len(predict_zero_position))


                    input_text = input_text.strip('[').strip(']').split(', ')
                    input_text_new = [Decimal(a).quantize(Decimal("0.001"), rounding="ROUND_HALF_UP") for a in input_text]
                    input_text = [float(a) for a in input_text_new]

                    input_text_result = []
                    if len(predict_zero_position) > 2:
                        for index in range(len(predict_zero_position)):
                            if predict_zero_position[index] < len(input_text) - 1:
                                if index != 0:
                                    input_text_result += input_text[
                                                         (predict_zero_position[index - 1] + 1):(
                                                         predict_zero_position[index])]
                                else:
                                    input_text_result += input_text[:predict_zero_position[0]]
                            elif predict_zero_position[index] == len(input_text) - 1:
                                input_text_result += input_text[(predict_zero_position[index - 1] + 1):-1]
                            else:
                                assert False
                    else:
                        if len(predict_zero_position) == 2:
                            input_text_result += input_text[:predict_zero_position[0]]
                            input_text_result += input_text[(predict_zero_position[0] + 1):-1]
                        elif len(predict_zero_position) == 1:
                            input_text_result += input_text[:-1]
                        else:
                            assert False

                    assert len(input_text_result) == (len(input_text) - len(predict_zero_position))


                    assert len(input_text_result) == len(predict_text_result)
                    if len(predict_text_result) >= 3072:
                        predict_text = predict_text_result[:3072]
                        input_text = input_text_result[:3072]
                    else:
                        division = int(3072 / len(predict_text_result)) + 1
                        predict_text = []
                        input_text = []
                        for i in range(division):
                            predict_text += predict_text_result
                            input_text += input_text_result
                        predict_text = predict_text[:3072]
                        input_text = input_text[:3072]
                    assert len(input_text) == len(predict_text) == 3072

                    # todo 验证input_text和predict_text的正确性

                    logits_difference_result = [a - b for a, b in zip(predict_text, input_text)]

                    logits_difference_input = [predict_text, input_text, logits_difference_result]
                    logits_difference_input = torch.tensor(logits_difference_input, dtype=torch.float)

                    data.dict_ = {'logits_difference_input': logits_difference_input, 'entities': key_list, 'message': message_text, "pub_time": pub_time, "title": title, "url": url, "content": content}

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
