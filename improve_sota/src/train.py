from model.MyBert import MyBert_V1,MyBert_V2,MyXLNet
# from MyDataset import MyDataset_V1
from create_graph import MyOwnDataset
import torch
from torch.nn import ReLU
import torch.nn.functional as F
import numpy as np
# from torch.utils.data import DataLoader
from torch import nn
import math
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
#from torchcontrib.optim import SWA
from torch.autograd import Variable
from AttackTraining import FGM,EMA,CosineAnnealingLRWarmup
from sklearn.model_selection import KFold
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--BATCH_SIZE', type=int, help='BATCH_SIZE', default=8)
parser.add_argument('--EPOCH', type=int, help='EPOCH', default=1)
parser.add_argument('--is_prompt', type=bool, help='BATCH_SIZE', default=False)
parser.add_argument('--is_mix_up', type=bool, help='is_mix_up', default=False)
parser.add_argument('--is_fgm', type=bool, help='is_fgm', default=False)
parser.add_argument('--DEVICE', type=str, help='DEVICE', default="cuda:0")
parser.add_argument('--model_name', type=str, help='model_name', default="nezha_base_cn")
parser.add_argument('--gamma', type=float, help='the gamma value of focalloss function', default=1.1)
parser.add_argument('--k_fold', type=int, help='The k value of k_fold', default=10)
parser.add_argument('--class_num', type=int, help='num of different classes', default=2)
parser.add_argument('--lr', type=float, help='learning rate', default=2e-5)
parser.add_argument('--seed', type=int, help='random seed', default=1997)
parser.add_argument('--max_seq_length', type=int, help='max_seq_length', default=512)
parser.add_argument('--vocab_size', type=int, help='vocab_size', default=21128)
parser.add_argument('--fold_list', type=int, nargs='+', help='fold_list', default=None)
parser.add_argument('--output_prefix', type=str, help='output_prefix', default="model_B_")
parser.add_argument('--all_data', type=bool, help='Is or not use all data for training', default=False)
parser.add_argument('--input_format', type=str,  help='input_format', required = True) # 指输入数据的格式，目前用到的有“statistics+320+64”、“statistics+512+128”、“200+entities”三种。第一种表示输入统计信息、标题、内容前320个字符和内容后64个字符；第二种表示输入统计信息、标题、内容前512个字符和内容后128个字符；最后一种表示输入标题、内容前200个字符和实体名称。
args = parser.parse_args()
DEVICE = args.DEVICE
SEED = args.seed
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.softmax(inputs,dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    

    
def acc(preds,targs,th=0.0):
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()

class F1_Score:
    def __init__(self,n=8):
        self.n = n
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def __call__(self,preds,targs,th=0.0):
        preds = (preds > th).int()
        targs = targs.int()
        self.TP += np.multiply(preds,targs).sum()
        self.FP += (preds > targs).sum()
        self.FN += (preds < targs).sum()
        score = (2.0*self.TP/(2.0*self.TP + self.FP + self.FN + 1e-8))
        return score

    def reset(self):
        #macro F1 score
        score = (2.0*self.TP/(2.0*self.TP + self.FP + self.FN + 1e-8))
        print('F1 macro:',score.mean(),flush=True)
        self.TP = 0
        self.FP = 0
        self.FN = 0

def build_batch_inputs(batch, is_prompt=False, return_y=False, whether_test=False):
    item = batch.dict_['origin_tuple']
    x_1 = [item[i][0] for i in range(len(item))]
    y = torch.tensor([item[i][-1] for i in range(len(item))])
    y = y.squeeze(-1)
    y = y.to(DEVICE)
    if return_y == True:
        return y
    if is_prompt == True and whether_test == False:
        input_ids = torch.tensor(np.array([x_1[i][0].detach().cpu().numpy() for i in range(len(x_1))]))
        masked_attention = torch.tensor(np.array([x_1[i][1].detach().cpu().numpy() for i in range(len(x_1))]))
        masked_index = torch.tensor(np.array([x_1[i][2].detach().cpu().numpy() for i in range(len(x_1))]))
        original_input = {"input_ids": input_ids.to(DEVICE), "attention_mask": masked_attention.to(DEVICE), "masked_index": masked_index.to(DEVICE)}
    elif is_prompt == False and whether_test == False:
        input_ids = torch.tensor(np.array([x_1[i][0].detach().cpu().numpy() for i in range(len(x_1))]))
        masked_attention = torch.tensor(np.array([x_1[i][1].detach().cpu().numpy() for i in range(len(x_1))]))
        original_input = {"input_ids": input_ids.to(DEVICE), "attention_mask": masked_attention.to(DEVICE)}
    elif is_prompt == True and whether_test == True:
        input_ids = torch.tensor(np.array([x_1[i][0].detach().cpu().numpy() for i in range(len(x_1))]))
        masked_attention = torch.tensor(np.array([x_1[i][1].detach().cpu().numpy() for i in range(len(x_1))]))
        masked_index = torch.tensor(np.array([x_1[i][2].detach().cpu().numpy() for i in range(len(x_1))]))
        original_input = {"input_ids": input_ids.view(1, -1).to(DEVICE), "attention_mask": masked_attention.view(1, -1).to(DEVICE), "masked_index": torch.LongTensor([masked_index]).view(-1).to(DEVICE)}
    elif is_prompt == False and whether_test == True:
        input_ids = torch.tensor(np.array([x_1[i][0].detach().cpu().numpy() for i in range(len(x_1))]))
        masked_attention = torch.tensor(np.array([x_1[i][1].detach().cpu().numpy() for i in range(len(x_1))]))
        original_input = {"input_ids": input_ids.view(1, -1).to(DEVICE), "attention_mask": masked_attention.view(1, -1).to(DEVICE)}
    else:
        assert False

    batch = batch.to(DEVICE)
    batch_tensor = batch['father'].batch.to(DEVICE)
    inputs = {"dict_": batch.dict_, "old_dict": batch.x_dict, "edge_index_dict": batch.edge_index_dict, "batch": batch_tensor}
    for key, value in inputs.items():
        original_input[key] = value

    return original_input


def compute_kl_loss(predict, target):
    pt_kl = F.kl_div(F.log_softmax(predict, dim=-1), F.softmax(target, dim=-1), reduction='mean')
    pp_kl = 0
    for length in range(len(target)):
        predict_new = predict[length]
        temp = F.kl_div(F.log_softmax(predict_new, dim=-1), F.softmax(predict, dim=-1), reduction='mean')
        pp_kl += temp
    pp_kl /= len(target)
    action = ReLU()
    loss = action(pt_kl - pp_kl)
    return loss

def loss_fn(pre, y):
    loss_fn_main = None
    if args.is_prompt:
        loss_fn_main = FocalLoss(class_num=args.vocab_size, gamma=args.gamma)
    else:
        loss_fn_main = FocalLoss(class_num=args.class_num, gamma=args.gamma)
    logits = pre['logits']
    gnn = pre["gnn"]
    lm = pre["lm"]
    final_loss = loss_fn_main(logits, y) + compute_kl_loss(gnn, lm)
    return final_loss

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    # 为当前GPU设置种子用于生成随机数，以使结果是确定的
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    kf = KFold(n_splits = args.k_fold,shuffle = True,random_state = args.seed)
    dataset = MyOwnDataset(root='train', modify_dataset = True,max_seq_length = args.max_seq_length,prompt = args.is_prompt,input_format = args.input_format,model_name = args.model_name)
    
    # dataloader = DataLoader(dataset,batch_size=args.BATCH_SIZE,drop_last = True,shuffle = True)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.BATCH_SIZE, drop_last=True)

    #loss_fn = torch.nn.CrossEntropyLoss()
    
    data_induce = np.arange(0, len(dataset))
    # 开始k折训练
    for fold,(train_index, val_index) in enumerate(kf.split(data_induce)):
        # 若指定了fold_list，则只训练指定的fold。
        if args.fold_list != None and fold not in args.fold_list:
            continue
        train_subset = torch.utils.data.dataset.Subset(dataset, train_index)
        val_subset = torch.utils.data.dataset.Subset(dataset, val_index)
        data_loaders = {}
        data_loaders['train'] = DataLoader(train_subset, batch_size = args.BATCH_SIZE, sampler=RandomSampler(train_subset))
        data_loaders['val'] = DataLoader(val_subset, batch_size = 1)
        print(len(data_loaders['train']))
        print(len(data_loaders['val']))
        if args.is_prompt:  
            model = MyBert_V3()
            state_dict = torch.load(f'trained_model/pretrained_{args.model_name}.pth')
            model.load_state_dict(state_dict)
        else:
            if args.model_name == "nezha-base-cn":
                model = MyBert_V1()
            elif args.model_name == "xlnet-base-cn":
                model = MyXLNet()
            else:
                raise NameError("未找到该模型，请重新填写或配置模型名！")
            

        model = model.to(DEVICE)
        
        def adjust_learning_rate(optimizer, current_step,max_step,lr_min=0,lr_max=1e-4,warmup=True):
            warmup_step = 100 if warmup else 0
            if current_step < warmup_step:
                lr = lr_max * (current_step / warmup_step)
            else:
                #lr = lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * (current_step - warmup_step) / (max_step - warmup_step))) / 2
                lr = args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        fgm = FGM(model)
        optimizer = torch.optim.Adam(lr=2e-5,params=model.parameters())
        ema = EMA(model, 0.9)
        ema.register()
        lr_min = args.lr / 2
        lr_max = args.lr * 2
        #optimizer = SWA(base_opt, swa_start=800, swa_freq=50, swa_lr=1e-5)
        cnt = 0
        for epoch in range(args.EPOCH):
            acc_val = 0
            loss_val = 0
            f1_score = F1_Score(args.BATCH_SIZE)
            model.train()
            if not args.all_data:
                dataloader = data_loaders['train']
            for item in dataloader:
                y = build_batch_inputs(batch=item, return_y=True)
                adjust_learning_rate(optimizer=optimizer,current_step=cnt,max_step=len(dataloader) * args.EPOCH,lr_min=lr_min,lr_max=lr_max,warmup=True)
                optimizer.zero_grad()
                model.zero_grad()

                if args.is_mix_up:
                    index = torch.randperm(y.shape[0])
                    lam = np.random.beta(0.2,0.2)
                    def single_forward_hook(module, inputs, outputs):
                        mix_input = outputs * lam + outputs[index] * (1 - lam)
                        return mix_input
                    def multi_forward_hook(module, inputs, outputs):
                        mix_input = outputs[0] * lam + outputs[0][index] * (1 - lam)
                        return tuple([mix_input])
                    hook = model.pretrained_model.encoder.layer[11].register_forward_hook(multi_forward_hook)
                    y_pre = model(batch=item)
                    loss = lam * loss_fn(y_pre,y) + (1 - lam) * loss_fn(y_pre,y[index])
                    hook.remove()

                else:
                    if args.is_prompt:
                        y_pre = model(**(build_batch_inputs(batch=item, is_prompt=True)))
                    else:
                        y_pre = model(**(build_batch_inputs(batch=item)))
                    loss = loss_fn(y_pre,y)
               
                loss.mean().backward()
                if args.is_fgm:
                    fgm.attack()
                    y_pre_adv = model(**(build_batch_inputs(batch=item)))
                    loss_adv = loss_fn(y_pre_adv,y)
                    loss_adv.mean().backward()
                    fgm.restore()

                loss_val += loss.mean().item()
                optimizer.step()
                ema.update()
                cnt += 1
                if cnt % 200 == 0:
                    print(f"step: {cnt}, loss: {loss_val / 200}")
                    loss_val = 0
            #optimizer.swap_swa_sgd()
            print(f"第 {epoch + 1} 轮训练结束")
            model.eval()
            ema.apply_shadow()
            for item in data_loaders['val']:
                y_dev = build_batch_inputs(batch=item, return_y=True)
                y_dev = y_dev.cpu()
                if args.is_prompt:
                    y_pre_dev = model(**(build_batch_inputs(batch=item, is_prompt=True, whether_test=True)))
                else:
                    y_pre_dev = model(**(build_batch_inputs(batch=item, whether_test=True)))
                logits = y_pre_dev['logits'].argmax(1).cpu()
                acc_val += acc(logits,y_dev.cpu())
                f1 = f1_score(logits,y_dev.cpu())
                
            print(f"acc: {acc_val/len(data_loaders['val'])}, f1: {f1}")
            acc_val = 0
            f1_score.reset()

            model.train()
            ema.restore()
            # 记得提前新建trained_model这一目录
        if args.all_data:
            torch.save(model.state_dict(),f"trained_model/{args.output_prefix}_all.pth")
        else:
            torch.save(model.state_dict(),f"trained_model/{args.output_prefix}_{fold}.pth")
        
        torch.cuda.empty_cache()
        if args.all_data:
            break
    print("训练结束")
