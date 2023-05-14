import torch
from transformers import XLNetForSequenceClassification, XLNetConfig
from torch.nn import Module
from torch import nn
import NEZHA.utils as utils
import math
import numpy as np
from NEZHA.modeling_nezha import BertModel,BertConfig,BertForMaskedLM
from transformers import BertForMaskedLM as RealBertForMaskedLM
# from transformers import BertModel as RealBertModel
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, SAGEConv, Linear, HGTConv
import pkuseg
from torch.nn import MultiheadAttention
SEED = 1997
torch.manual_seed(SEED)
    # 为当前GPU设置种子用于生成随机数，以使结果是确定的
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


seg = pkuseg.pkuseg()
word2id = {}
weight_list = []
with open('./data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', 'r', encoding='utf-8') as vector_file:
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


class MyBert_V1(Module):
    def __init__(self,hidden_size = 768, model_name='nezha-base-cn'):
        super(MyBert_V1,self).__init__()
        self.pooler = nn.Linear(hidden_size,hidden_size)
        self.Tanh = nn.Tanh()

        # todo -------------original-------------------
        # self.classification = nn.Linear(hidden_size,2)
        # todo -------------original-------------------


        self.dropout = nn.Dropout(0.2)
        self.config = BertConfig("NEZHA/bert_config.json")
        self.pretrained_model = BertModel(config=self.config)
        utils.torch_init_model(self.pretrained_model,'NEZHA/pytorch_model.bin')


        self.device = 'cuda:0'
        self.model_name = model_name
        state_dict = {}
        original_state_dict = torch.load(f'trained_model/pretrained_{self.model_name}.pth')
        for key in original_state_dict:
            val = original_state_dict[key]
            new_key = key.replace(".bert", "")
            state_dict[new_key] = val
        state_dict = original_state_dict
        self.pretrained_model.load_state_dict(state_dict, strict=False)



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
            ("entities", "has", "message"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                    heads=self.heads_first,
                                                    add_self_loops=False),
            ("message", "belong", "entities"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                       heads=self.heads_first, add_self_loops=False),
            ("entities", "co_occurrence", "entities"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                               heads=self.heads_first, add_self_loops=False),
            ("entities", "refer", "father"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                     heads=self.heads_first,
                                                     add_self_loops=False),
        }, aggr='sum')
        self.convs.append(conv)
        conv = HeteroConv({
            ("entities", "has", "message"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                    out_channels=self.out_channels, heads=self.heads_second,
                                                    add_self_loops=False),
            ("message", "belong", "entities"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                       out_channels=self.out_channels,
                                                       heads=self.heads_second, add_self_loops=False),
            ("entities", "co_occurrence", "entities"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                               out_channels=self.out_channels,
                                                               heads=self.heads_second, add_self_loops=False),
            ("entities", "refer", "father"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                     out_channels=self.out_channels, heads=self.heads_second,
                                                     add_self_loops=False),
        }, aggr='sum')
        self.convs.append(conv)

        # logits_difference部分, 输入为3个2048维向量
        self.logits_difference_input_dim = 2048
        self.attention_heads = 8
        self.query_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.key_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.value_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.muti_head_attention = MultiheadAttention(embed_dim=self.logits_difference_input_dim, num_heads=self.attention_heads, dropout=0.1, batch_first=True)
        self.logits_difference_dimensionality_reduction_trans_mlp = nn.ModuleList([nn.Linear(self.logits_difference_input_dim, int(self.logits_difference_input_dim / self.attention_heads)) for i in range(3)])
        self.logits_difference_mlp = nn.Linear(3 * int(self.logits_difference_input_dim / self.attention_heads), int(self.logits_difference_input_dim / self.attention_heads))

        # 融合部分
        self.mlp_hidden_size = 896
        self.MLP_1 = nn.Linear(self.config.hidden_size + self.out_channels + int(self.logits_difference_input_dim / self.attention_heads), self.mlp_hidden_size, bias=True)
        self.MLP_2 = nn.Linear(self.mlp_hidden_size, self.config.num_labels, bias=True)
        nn.init.normal_(self.MLP_1.weight, std=0.01)
        nn.init.normal_(self.MLP_1.bias, std=0.01)
        nn.init.normal_(self.MLP_2.weight, std=0.01)
        nn.init.normal_(self.MLP_2.bias, std=0.01)
        self.act = nn.LeakyReLU()

    def forward(self,input_ids,attention_mask, dict_, old_dict, edge_index_dict, batch):

        output = self.pretrained_model(input_ids, attention_mask=attention_mask)[0]
        # print(output.shape)
        pooled_output = self.pooler(output)
        # print(pooled_output.shape)
        pooled_output = self.Tanh(pooled_output)
        # print(pooled_output.shape)
        pooled_output = torch.mean(pooled_output, 1)
        # print(pooled_output.shape)
        pooled_output = self.dropout(pooled_output)
        # print(pooled_output.shape)




        # todo -------------original-------------------
        # output = self.classification(pooled_output)
        # todo -------------original-------------------


        url = dict_['url']
        key_list = dict_['entities']
        message_text = dict_['message']
        logits_difference_input = dict_['logits_difference_input']   # (batch_size, 3, 2048)

        # gnn部分
        key_tensor = []
        message_tensor = []
        # 获取知识图谱的向量表示
        batch_size = len(url)
        for batch_iter in range(batch_size):
            for item in key_list[batch_iter]:
                new_tensor = np.array([0.0] * 300)
                for word in seg.cut(item):
                    if word not in word2id.keys():
                        word = '，'
                    new_tensor += np.array(
                        self.embedding(torch.LongTensor([self.word2id[word]]).to(self.device)).detach().cpu()).squeeze()
                key_tensor.append(new_tensor.tolist())

            for item in message_text[batch_iter]:
                new_tensor = np.array([0.0] * 300)
                for word in seg.cut(item):
                    if word not in word2id.keys():
                        word = '，'
                    new_tensor += np.array(
                        self.embedding(torch.LongTensor([self.word2id[word]]).to(self.device)).detach().cpu()).squeeze()
                message_tensor.append(new_tensor.tolist())

        new_dict = {'entities': torch.tensor(np.array(key_tensor), dtype=torch.float).to(self.device),
                    'message': torch.tensor(np.array(message_tensor), dtype=torch.float).to(self.device),
                    'father': torch.tensor([[0.0] * 300] * batch_size, dtype=torch.float).to(self.device),
                    }
        for key, value in new_dict.items():
            old_dict[key] = new_dict[key]
        # https://github.com/pyg-team/pytorch_geometric/issues/2844
        for conv in self.convs:
            old_dict = conv(old_dict, edge_index_dict)

        # logits_difference部分
        logits_difference_input = torch.tensor(logits_difference_input, dtype=torch.float).to(self.device)
        query = self.query_linear(logits_difference_input)
        key = self.key_linear(logits_difference_input)
        value = self.value_linear(logits_difference_input)
        # todo 构建attention_mask
        logits_difference_input, _ = self.muti_head_attention(query, key, value)
        logits_difference_input = logits_difference_input.transpose(0, 1)
        logits_difference_tmp = torch.zeros((3, len(url), int(self.logits_difference_input_dim / self.attention_heads)))
        logits_difference_tmp = torch.tensor(logits_difference_tmp, dtype=torch.float).to(self.device)
        for i, model_item in enumerate(self.logits_difference_dimensionality_reduction_trans_mlp):
            logits_difference_tmp[i] = model_item(logits_difference_input[i])
        logits_difference_input = logits_difference_tmp
        logits_difference_input = logits_difference_input.transpose(0, 1)
        logits_difference_input = logits_difference_input.reshape(-1, int(self.logits_difference_input_dim / self.attention_heads) * 3)
        logits_difference_input = self.logits_difference_mlp(logits_difference_input)

        # 融合部分
        hidden_tensor = torch.cat([logits_difference_input, pooled_output, old_dict["father"]], dim=1)
        logits = self.MLP_2(self.act(self.MLP_1(hidden_tensor)))
        return {"logits": logits, "gnn": old_dict["father"], "lm": pooled_output}

    
    
class MyBert_V2(Module):
    def __init__(self,hidden_size = 768,model_name = "nezha-base-cn"):
        super(MyBert_V2,self).__init__()
        self.model_name = model_name
        if model_name == "bert-base-chinese":
            self.pretrained_model = RealBertForMaskedLM.from_pretrained("bert-base-chinese")
        elif model_name == "nezha-base-cn":
            self.config = BertConfig("NEZHA/bert_config.json")
            self.pretrained_model = BertForMaskedLM(config=self.config)
            utils.torch_init_model(self.pretrained_model,'NEZHA/pytorch_model.bin')
        else:
            raise NameError("未找到该模型，请检查或配置该模型")
    def forward(self,input_ids,attention_mask):
        if self.model_name == "bert-base-chinese":
            prediction_scores = self.pretrained_model(input_ids,attention_mask=attention_mask)[0]
        else:
            prediction_scores = self.pretrained_model(input_ids,attention_mask=attention_mask)
        output = prediction_scores.view(-1, 21128)
        return output


class MyXLNet(nn.Module):
    def __init__(self,num_labels = 2, max_seq_len=768):
        super(MyXLNet,self).__init__()
        self.config = XLNetConfig.from_pretrained("hfl/chinese-xlnet-base")
        self.config.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.bert = XLNetForSequenceClassification.from_pretrained("hfl/chinese-xlnet-base",config = self.config)


        self.device = 'cuda:0'
        self.max_length = max_seq_len
        self.pooler = nn.Linear(self.max_length, self.max_length)
        # self.Tanh = nn.Tanh()
        # gnn部分
        self.word2id = word2id
        # todo embedding的padding_idx是否可以采用-100
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_list), freeze=False, padding_idx=-100)
        self.num_layers = 2
        self.out_channels = self.max_length
        self.heads_first = 4
        self.heads_second = 1
        self.convs = torch.nn.ModuleList()
        self.add_self_loops = False
        conv = HeteroConv({
            ("entities", "has", "message"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                    heads=self.heads_first,
                                                    add_self_loops=False),
            ("message", "belong", "entities"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                       heads=self.heads_first, add_self_loops=False),
            ("entities", "co_occurrence", "entities"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                               heads=self.heads_first, add_self_loops=False),
            ("entities", "refer", "father"): GATConv(in_channels=300, out_channels=self.out_channels,
                                                     heads=self.heads_first,
                                                     add_self_loops=False),
        }, aggr='sum')
        self.convs.append(conv)
        conv = HeteroConv({
            ("entities", "has", "message"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                    out_channels=self.out_channels, heads=self.heads_second,
                                                    add_self_loops=False),
            ("message", "belong", "entities"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                       out_channels=self.out_channels,
                                                       heads=self.heads_second, add_self_loops=False),
            ("entities", "co_occurrence", "entities"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                               out_channels=self.out_channels,
                                                               heads=self.heads_second, add_self_loops=False),
            ("entities", "refer", "father"): GATConv(in_channels=self.heads_first * self.out_channels,
                                                     out_channels=self.out_channels, heads=self.heads_second,
                                                     add_self_loops=False),
        }, aggr='sum')
        self.convs.append(conv)

        # logits_difference部分, 输入为3个2048维向量
        self.logits_difference_input_dim = 2048
        self.attention_heads = 8
        self.query_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.key_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.value_linear = nn.Linear(self.logits_difference_input_dim, self.logits_difference_input_dim)
        self.muti_head_attention = MultiheadAttention(embed_dim=self.logits_difference_input_dim,
                                                      num_heads=self.attention_heads, dropout=0.1, batch_first=True)
        self.logits_difference_dimensionality_reduction_trans_mlp = nn.ModuleList(
            [nn.Linear(self.logits_difference_input_dim, int(self.logits_difference_input_dim / self.attention_heads))
             for i in range(3)])
        self.logits_difference_mlp = nn.Linear(3 * int(self.logits_difference_input_dim / self.attention_heads),
                                               int(self.logits_difference_input_dim / self.attention_heads))

        # 融合部分
        self.mlp_hidden_size = int((self.max_length + self.out_channels + int(self.logits_difference_input_dim / self.attention_heads)) / 2)
        self.MLP_1 = nn.Linear(
            self.max_length + self.out_channels + int(self.logits_difference_input_dim / self.attention_heads),
            self.mlp_hidden_size, bias=True)
        self.MLP_2 = nn.Linear(self.mlp_hidden_size, self.config.num_labels, bias=True)
        nn.init.normal_(self.MLP_1.weight, std=0.01)
        nn.init.normal_(self.MLP_1.bias, std=0.01)
        nn.init.normal_(self.MLP_2.weight, std=0.01)
        nn.init.normal_(self.MLP_2.bias, std=0.01)
        self.act = nn.LeakyReLU()


    def forward(self,input_ids,attention_mask, dict_, old_dict, edge_index_dict, batch):

        # todo -------------original-------------------
        # y = self.bert(input_ids,attention_mask=attention_mask)[0]
        # y = self.dropout(y)
        # todo -------------original-------------------



        # # todo 使用最后一层的cls
        # output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:, 0, :]
        # todo 还可以采用对cls进行pooler
        output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1][:, 0, :]
        pooled_output = self.pooler(output)
        # # todo 还可以采用像Bert_1那样，对输出最后一层进行pooler然后取mean再dropout
        # output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        # pooled_output = self.pooler(output)
        # pooled_output = self.Tanh(pooled_output)
        # pooled_output = torch.mean(pooled_output, 1)
        # pooled_output = self.dropout(pooled_output)



        url = dict_['url']
        key_list = dict_['entities']
        message_text = dict_['message']
        logits_difference_input = dict_['logits_difference_input']  # (batch_size, 3, 2048)

        # gnn部分
        key_tensor = []
        message_tensor = []
        # 获取知识图谱的向量表示
        batch_size = len(url)
        for batch_iter in range(batch_size):
            for item in key_list[batch_iter]:
                new_tensor = np.array([0.0] * 300)
                for word in seg.cut(item):
                    if word not in word2id.keys():
                        word = '，'
                    new_tensor += np.array(
                        self.embedding(torch.LongTensor([self.word2id[word]]).to(self.device)).detach().cpu()).squeeze()
                key_tensor.append(new_tensor.tolist())

            for item in message_text[batch_iter]:
                new_tensor = np.array([0.0] * 300)
                for word in seg.cut(item):
                    if word not in word2id.keys():
                        word = '，'
                    new_tensor += np.array(
                        self.embedding(torch.LongTensor([self.word2id[word]]).to(self.device)).detach().cpu()).squeeze()
                message_tensor.append(new_tensor.tolist())

        new_dict = {'entities': torch.tensor(np.array(key_tensor), dtype=torch.float).to(self.device),
                    'message': torch.tensor(np.array(message_tensor), dtype=torch.float).to(self.device),
                    'father': torch.tensor([[0.0] * 300] * batch_size, dtype=torch.float).to(self.device),
                    }
        for key, value in new_dict.items():
            old_dict[key] = new_dict[key]
        # https://github.com/pyg-team/pytorch_geometric/issues/2844
        for conv in self.convs:
            old_dict = conv(old_dict, edge_index_dict)

        # logits_difference部分
        logits_difference_input = torch.tensor(logits_difference_input, dtype=torch.float).to(self.device)
        query = self.query_linear(logits_difference_input)
        key = self.key_linear(logits_difference_input)
        value = self.value_linear(logits_difference_input)
        # todo 构建attention_mask
        logits_difference_input, _ = self.muti_head_attention(query, key, value)
        logits_difference_input = logits_difference_input.transpose(0, 1)
        # logits_difference_input = self.logits_difference_no_dimensionality_reduction_trans_mlp(logits_difference_input)
        logits_difference_tmp = torch.zeros((3, len(url), int(self.logits_difference_input_dim / self.attention_heads)))
        logits_difference_tmp = torch.tensor(logits_difference_tmp, dtype=torch.float).to(self.device)
        for i, model_item in enumerate(self.logits_difference_dimensionality_reduction_trans_mlp):
            logits_difference_tmp[i] = model_item(logits_difference_input[i])
        logits_difference_input = logits_difference_tmp
        logits_difference_input = logits_difference_input.transpose(0, 1)
        logits_difference_input = logits_difference_input.reshape(-1,
                                                                  int(self.logits_difference_input_dim / self.attention_heads) * 3)
        logits_difference_input = self.logits_difference_mlp(logits_difference_input)

        # 融合部分
        hidden_tensor = torch.cat([logits_difference_input, pooled_output, old_dict["father"]], dim=1)
        logits = self.MLP_2(self.act(self.MLP_1(hidden_tensor)))

        return {"logits": logits, "gnn": old_dict["father"], "lm": pooled_output}


