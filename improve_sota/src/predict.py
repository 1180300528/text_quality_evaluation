from model.MyBert import MyBert_V1,MyXLNet
# from MyDataset import MyDataset_V1
from create_graph import MyOwnDataset
import torch
import numpy as np
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int, help='BATCH_SIZE', default=8)
parser.add_argument('--DEVICE', type=str, help='DEVICE', default="cuda:0")
parser.add_argument('--model_name', type=str, help='model_name', default="nezha_base_cn")
parser.add_argument('--class_num', type=int, help='num of different classes', default=2)
parser.add_argument('--max_seq_length', type=int, help='max_seq_length', default=512)
parser.add_argument('--model_list', type=str, nargs='+', help='model_list', default=None)
parser.add_argument('--input_format', type=str,  help='input_format', required = True) # 指输入数据的格式，目前用到的有“statistics+320+64”、“statistics+512+256”、“200+entities”三种。第一种表示输入统计信息、标题、内容前320个字符和内容后64个字符；第二种表示输入统计信息、标题、内容前512个字符和内容后256个字符；最后一种表示输入标题、内容前200个字符和实体名称。
args = parser.parse_args()
DEVICE = args.DEVICE

def build_batch_inputs(batch, return_url=False, return_last=False):
    item = batch.dict_['origin_tuple']
    url = [item[i][0] for i in range(len(item))]
    x_1 = [item[i][1] for i in range(len(item))]
    x_2 = [item[i][2] for i in range(len(item))]
    if return_url == True:
        return url
    elif return_last == True:
        return x_2
    elif return_last == False:
        input_ids = torch.tensor(np.array([x_1[i][0].detach().cpu().numpy() for i in range(len(x_1))]))
        masked_attention = torch.tensor(np.array([x_1[i][1].detach().cpu().numpy() for i in range(len(x_1))]))
        original_input = {"input_ids": input_ids.to(DEVICE), "attention_mask": masked_attention.to(DEVICE)}
    else:
        assert False

    batch = batch.to(DEVICE)
    batch_tensor = batch['father'].batch.to(DEVICE)
    inputs = {"dict_": batch.dict_, "old_dict": batch.x_dict, "edge_index_dict": batch.edge_index_dict,
              "batch": batch_tensor}
    for key, value in inputs.items():
        original_input[key] = value

    return original_input

if __name__ == '__main__':
    dataset = MyOwnDataset(root='eval', modify_dataset = True,is_training = False,max_seq_length = args.max_seq_length,input_format = args.input_format,model_name = args.model_name)
    dataloader = DataLoader(dataset, batch_size=args.BATCH_SIZE)
    for model_name in args.model_list:
        if args.model_name == "nezha-base-cn":
            model = MyBert_V1()
        elif args.model_name == "xlnet-base-cn":
            model = MyXLNet()
        else:
            raise NameError("未找到该模型，请重新填写或配置模型名！")
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        s = ""
        cnt_0 = 0
        cnt_1 = 0
        for item in dataloader:
            url = build_batch_inputs(item, return_url=True)
            y_pre = model(**(build_batch_inputs(item)))
            y_pre = y_pre['logits']
            logits = list(map(int,list(torch.argmax(y_pre,1))))
            #print(y_pre)
            for i in range(0,len(url)):
                dic = dict()
                dic["url"] = url[i]
                dic["label"] = logits[i]
                if dic["label"] == 1:
                    cnt_1 += 1
                else:
                    cnt_0 += 1
                s += json.dumps(dic) + "\n"
        print(cnt_0,cnt_1)
        output_prefix = model_name.replace(".pth","")
        output_prefix = output_prefix.replace("trained_model/","")
        with open(f"output/result_{output_prefix}.txt",mode='w') as f:
            f.write(s)
            f.close()
        