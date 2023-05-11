import torch
from torchblocks.core import TrainerBase


class TextClassifierTrainer(TrainerBase):
    '''
    文本分类
    '''
    def build_batch_concat(self, all_batch_list, dim=0):
        if self.opts.loss_type is not None:
            if self.opts.loss_type == 'cross':
                pass
            if self.opts.loss_type == 'MutiChoice':
                for batch in all_batch_list:
                    batch["logits"] = torch.where(batch["logits"] > 0, 1, 0)
            if self.opts.loss_type  == "BCE":
                for batch in all_batch_list:
                    batch['logits'] = torch.sigmod(batch['logits'])
                    batch["logits"] = torch.where(batch["logits"] > 0.5, 1, 0)

        preds = torch.cat([batch['logits'] for batch in all_batch_list], dim=dim)
        target = torch.cat([batch['label'].int() for batch in all_batch_list], dim=dim)
        return {"preds": preds, "target": target}

