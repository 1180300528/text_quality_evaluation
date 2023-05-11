import numpy
import torch
from torch.nn import MultiheadAttention
import torch.nn as nn
import numpy as np
import copy
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, SAGEConv, Linear, HGTConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from transformers import BertModel, BertForSequenceClassification
from torch.nn.functional import cross_entropy
import pkuseg
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.batch import Batch
from torch.nn.utils.rnn import pad_sequence
from loss import compute_kl_loss


seg = pkuseg.pkuseg()
word2id = {}
weight_list = []
with open('../../../data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', 'r', encoding='utf-8') as vector_file:
    lines = vector_file.readlines()[1:]
    for index, line in enumerate(lines):
        list_ = line.split(' ')[:-1]
        try:
            assert len(list_) == 301
        except Exception as e:
            print(len(list_))
            print(list)
            assert False
        word2id[list_[0]] = index
        weight_list.append([float(i) for i in list_[1:]])


class HeteroGNN_with_RoBERTa(torch.nn.Module):
    def __init__(self, model_name_or_path, config, tokenizer, device):
        super().__init__()
        self.device = device
        # roberta部分
        self.encoder = BertForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        # self.dropout = nn.Dropout(0.2)
        self.tokenizer = tokenizer

        # gnn部分
        self.word2id = word2id
        # todo embedding的padding_idx是否可以采用-100
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_list), freeze=False, padding_idx=-100)
        self.num_layers = 2
        self.out_channels = 768
        self.heads_first = 4
        self.heads_second = 1
        self.convs = torch.nn.ModuleList()
        self.add_self_loops = False
        conv = HeteroConv({
            ("entities", "has", "message"): GATConv(in_channels=300, out_channels=self.out_channels, heads=self.heads_first,
                                                    add_self_loops=False),
            ("message", "belong", "entities"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                       heads=self.heads_first, add_self_loops=False),
            ("entities", "co_occurrence", "entities"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                               heads=self.heads_first, add_self_loops=False),
            # ("url", "refer", "father"): SAGEConv(in_channels=300, out_channels=self.heads_first * self.out_channels),
            ("entities", "refer", "father"): GATConv(in_channels=300, out_channels=self.out_channels, heads=self.heads_first,
                                                     add_self_loops=False),
            # ("title", "refer", "father"): SAGEConv(in_channels=300, out_channels=self.heads_first * self.out_channels),
            # ("pub_time", "refer", "father"): SAGEConv(in_channels=300, out_channels=self.heads_first * self.out_channels),
        }, aggr='sum')
        self.convs.append(conv)
        conv = HeteroConv({
            ("entities", "has", "message"): GATConv(in_channels=self.heads_first * self.out_channels, out_channels=self.out_channels, heads=self.heads_second,
                                                    add_self_loops=False),
            ("message", "belong", "entities"): GATConv(in_channels=self.heads_first * self.out_channels, out_channels=self.out_channels,
                                                       heads=self.heads_second, add_self_loops=False),
            ("entities", "co_occurrence", "entities"): GATConv(in_channels=self.heads_first * self.out_channels, out_channels=self.out_channels,
                                                               heads=self.heads_second, add_self_loops=False),
            # ("url", "refer", "father"): SAGEConv(in_channels=self.heads_first * self.out_channels, out_channels=self.heads_second * self.out_channels),
            ("entities", "refer", "father"): GATConv(in_channels=self.heads_first * self.out_channels, out_channels=self.out_channels, heads=self.heads_second,
                                                     add_self_loops=False),
            # ("title", "refer", "father"): SAGEConv(in_channels=self.heads_first * self.out_channels, out_channels=self.heads_second * self.out_channels),
            # ("pub_time", "refer", "father"): SAGEConv(in_channels=self.heads_first * self.out_channels, out_channels=self.heads_second * self.out_channels),
        }, aggr='sum')
        self.convs.append(conv)
        self.pool = global_add_pool

        # logits_difference部分, 输入为3个3072维向量
        self.logits_difference_input_dim = 3072
        self.attention_heads = 12
        self.query_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.key_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.value_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.muti_head_attention = MultiheadAttention(embed_dim=self.logits_difference_input_dim, num_heads=self.attention_heads, dropout=0.1, batch_first=True)
        # self.logits_difference_no_dimensionality_reduction_trans_mlp = nn.ModuleList([nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim) for i in range(3)])
        self.logits_difference_dimensionality_reduction_trans_mlp = nn.ModuleList([nn.Linear(self.logits_difference_input_dim, int(self.logits_difference_input_dim / self.attention_heads)) for i in range(3)])
        self.logits_difference_mlp = nn.Linear(3 * int(self.logits_difference_input_dim / self.attention_heads), int(self.logits_difference_input_dim / self.attention_heads))

        # 融合部分
        self.mlp_hidden_size = 896
        self.MLP_1 = nn.Linear(config.hidden_size + self.out_channels + int(self.logits_difference_input_dim / self.attention_heads), self.mlp_hidden_size, bias=True)
        self.MLP_2 = nn.Linear(self.mlp_hidden_size, config.num_labels, bias=True)
        nn.init.normal_(self.MLP_1.weight, std=0.01)
        nn.init.normal_(self.MLP_1.bias, std=0.01)
        nn.init.normal_(self.MLP_2.weight, std=0.01)
        nn.init.normal_(self.MLP_2.bias, std=0.01)
        self.act = nn.ReLU()

    def forward(self, label, dict_, old_dict, edge_index_dict, batch):

        title = dict_['title']
        url = dict_['url']
        pub_time = dict_['pub_time']
        content = dict_['content']
        key_list = dict_['entities']
        message_text = dict_['message']
        logits_difference_input = dict['logits_difference_input']   # (batch_size, 3, 3072)

        for item in range(len(url)):
            content[item] = pub_time[item] + ' ' + title[item] + ' ' + content[item]

        # roberta部分
        inputs = self.tokenizer(content, return_tensors="pt", max_length=512, padding=True, truncation=True)
        bert_output = \
        self.encoder(inputs['input_ids'].cuda(self.device), inputs['attention_mask'].cuda(self.device),output_hidden_states=True).hidden_states[-1][:, 0, :]


        # # 全传入运行
        # length_dict = {}
        # max_length = 0
        # for i in range(len(url)):
        #     length_dict[i] = len(content[i])
        #     if max_length < len(content[i]):
        #         max_length = len(content[i])
        # for i in range(len(url)):
        #     if len(content[i]) < max_length:
        #         content[i] += ['[PAD]' * (max_length - len(content[i]))][0]
        # bert_output_dict = {}
        # block_number = min(int(max_length / 510) + 1, 4)
        # for block_item in range(block_number):
        #     content_need = [item[510 * block_item: 510 * (block_item + 1)] for item in content]
        #     inputs = self.tokenizer(content_need, return_tensors="pt", max_length=512, padding=True,
        #                             truncation=True)
        #     bert_output = \
        #         self.encoder(inputs['input_ids'].cuda(self.device), inputs['attention_mask'].cuda(self.device),
        #                      output_hidden_states=True).hidden_states[-1][:, 0, :]
        #     bert_output_dict[block_item] = bert_output
        # bert_output_final = [0.0] * len(url)
        # for single in range(len(url)):
        #     final_tensor = torch.FloatTensor([0.0] * 768).cuda(self.device)
        #     # print('final_tensor origin:')
        #     # print(final_tensor.shape)
        #     for sum_item in range(min(int(length_dict[single] / 510) + 1, 4)):
        #         final_tensor += bert_output_dict[sum_item][single]
        #         # print('final_tensor:')
        #         # print(final_tensor.shape)
        #     bert_output_final[single] = final_tensor.detach().cpu().numpy()
        # bert_output = torch.Tensor(bert_output_final).cuda(self.device)
        # # print(bert_output.shape)


        # gnn部分
        key_tensor = []
        message_tensor = []
        title_tensor = []
        url_tensor = []
        pub_time_tensor = []

        # # 获取知识图谱的向量表示
        # batch_size = len(url)
        # for batch_iter in range(batch_size):
        #     key_tensor_batch = []
        #     for item in key_list[batch_iter]:
        #
        #         new_tensor = np.array([0.0] * 300)
        #         for word in seg.cut(item):
        #
        #             if word not in word2id.keys():
        #                 word = '，'
        #             new_tensor += np.array(self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
        #
        #         key_tensor_batch.append(new_tensor)
        #     key_tensor.append(np.array(key_tensor_batch))
        #
        #     message_tensor_batch = []
        #     for item in message_text[batch_iter]:
        #         new_tensor = np.array([0.0] * 300)
        #         for word in seg.cut(item):
        #             if word not in word2id.keys():
        #                 word = '，'
        #             new_tensor += np.array(self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
        #         message_tensor_batch.append(new_tensor)
        #     message_tensor.append(np.array(message_tensor_batch))
        #     new_tensor = np.array([0.0] * 300)
        #     for word in seg.cut(title[batch_iter]):
        #         if word not in word2id.keys():
        #             word = '，'
        #         new_tensor += np.array(self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
        #     title_tensor.append(np.array([new_tensor]))
        #     new_tensor = np.array([0.0] * 300)
        #     for word in seg.cut(url[batch_iter]):
        #         if word not in word2id.keys():
        #             word = '，'
        #         new_tensor += np.array(self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
        #     url_tensor.append(np.array([new_tensor]))
        #     new_tensor = np.array([0.0] * 300)
        #     for word in seg.cut(pub_time[batch_iter]):
        #         if word not in word2id.keys():
        #             word = '，'
        #         new_tensor += np.array(self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
        #     pub_time_tensor.append(np.array([new_tensor]))



        # 获取知识图谱的向量表示
        batch_size = len(url)
        for batch_iter in range(batch_size):
            for item in key_list[batch_iter]:
                new_tensor = np.array([0.0] * 300)
                for word in seg.cut(item):
                    if word not in word2id.keys():
                        word = '，'
                    new_tensor += np.array(
                        self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
                key_tensor.append(new_tensor.tolist())

            for item in message_text[batch_iter]:
                new_tensor = np.array([0.0] * 300)
                for word in seg.cut(item):
                    if word not in word2id.keys():
                        word = '，'
                    new_tensor += np.array(
                        self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
                message_tensor.append(new_tensor.tolist())
            new_tensor = np.array([0.0] * 300)
            for word in seg.cut(title[batch_iter]):
                if word not in word2id.keys():
                    word = '，'
                new_tensor += np.array(
                    self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
            title_tensor.append(new_tensor.tolist())
            new_tensor = np.array([0.0] * 300)
            for word in seg.cut(url[batch_iter]):
                if word not in word2id.keys():
                    word = '，'
                new_tensor += np.array(
                    self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
            url_tensor.append(new_tensor.tolist())
            new_tensor = np.array([0.0] * 300)
            for word in seg.cut(pub_time[batch_iter]):
                if word not in word2id.keys():
                    word = '，'
                new_tensor += np.array(
                    self.embedding(torch.LongTensor([self.word2id[word]]).cuda(self.device)).detach().cpu()).squeeze()
            pub_time_tensor.append(new_tensor.tolist())

        new_dict = {'entities': torch.tensor(np.array(key_tensor), dtype=torch.float).cuda(self.device),
                    'message': torch.tensor(np.array(message_tensor), dtype=torch.float).cuda(self.device),
                    # 'title': torch.tensor(title_tensor, dtype=torch.float).cuda(self.device),
                    # 'url': torch.tensor(url_tensor, dtype=torch.float).cuda(self.device),
                    # 'pub_time': torch.tensor(pub_time_tensor, dtype=torch.float).cuda(self.device),
                    'father':torch.tensor([[0.0] * 300] * batch_size, dtype=torch.float).cuda(self.device),
                    }

        for key, value in new_dict.items():
            old_dict[key] = new_dict[key]
        # https://github.com/pyg-team/pytorch_geometric/issues/2844

        # breakpoint()
        for conv in self.convs:
            old_dict = conv(old_dict, edge_index_dict)
            # new_dict = {key: x.relu() for key, x in new_dict.items()}
        # 只对节点类型为needed_nodes_type的节点进行处理并返回
        # graph_output = self.pool(new_dict["father"], batch)

        # logits_difference部分
        logits_difference_input.cuda()
        query = self.query_linear(logits_difference_input)
        key = self.key_linear(logits_difference_input)
        value = self.value_linear(logits_difference_input)
        # todo 构建attention_mask
        logits_difference_input = self.muti_head_attention(query, key, value)
        # logits_difference_input = self.logits_difference_no_dimensionality_reduction_trans_mlp(logits_difference_input)
        logits_difference_input = self.logits_difference_dimensionality_reduction_trans_mlp(logits_difference_input)
        logits_difference_input = logits_difference_input.reshape(-1, int(self.logits_difference_input_dim / self.attention_heads) * 3)
        logits_difference_input = self.logits_difference_mlp(logits_difference_input)

        # 融合部分
        hidden_tensor = torch.cat([logits_difference_input, bert_output, old_dict["father"]], dim=1)
        logits = self.MLP_2(self.act(self.MLP_1(hidden_tensor)))
        loss = cross_entropy(input=logits, target=label) + compute_kl_loss(old_dict["father"], bert_output)

        if torch.isnan(loss):
            if torch.isnan(cross_entropy(input=logits, target=label)):
                print('交叉熵部分为nan')
                print(logits)
                print(bert_output.shape)
                print(bert_output)
                print(old_dict["father"].shape)
                print(old_dict["father"])
                print(hidden_tensor.shape)
                print(hidden_tensor)
            if torch.isnan(compute_kl_loss(old_dict["father"], bert_output)):
                print('kl散度部分为nan')
                print(logits)
                print(bert_output.shape)
                print(bert_output)
                print(old_dict["father"].shape)
                print(old_dict["father"])
                print(hidden_tensor.shape)
                print(hidden_tensor)
        return {"logits": logits, "loss": loss}
