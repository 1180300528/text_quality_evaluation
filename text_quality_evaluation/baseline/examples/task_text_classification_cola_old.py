import os
import csv
import json
import sys
import random
import torch
import torch.nn as nn
sys.path.append('..')
from typing import List, Dict, Callable, Any
from torchblocks.core import TextClassifierTrainer
from torchblocks.data.dataset import DatasetBase
from torchblocks.utils.seed import seed_everything
from torchblocks.utils.options import Argparser
from torchblocks.utils.device import prepare_device
from torchblocks.utils.logger import Logger
from torchblocks.utils.paths import check_dir
from torchblocks.data.process_base import ProcessBase
from torchblocks.utils.paths import find_all_checkpoints
from torchmetrics.classification import F1
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from transformers import BertModel, BertPreTrainedModel
from transformers import XLNetConfig, XLNetTokenizer, XLNetForSequenceClassification, XLNetModel


# todo 注意！！！max_len要与opt参数设置一致！！！
class MutilLabelClassification(BertPreTrainedModel):
    def __init__(self,config,max_len=128):
        super(MutilLabelClassification,self).__init__(config)
        self.max_len = max_len
        self.bert = BertModel(config=config)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)

    def forward(self,inputs):
        output = self.bert(**inputs,return_dict=True, output_hidden_states=True)
        #采用最后一层
        embedding = output.hidden_states[-1]
        embedding = self.pooling(embedding,inputs)
        output = self.classifier(embedding)
        return output

    def pooling(self,token_embeddings,input):
        output_vectors = []
        #attention_mask
        attention_mask = input['attention_mask']
        #[B,L]------>[B,L,1]------>[B,L,768],矩阵的值是0或者1
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        #这里做矩阵点积，就是对元素相乘(序列中padding字符，通过乘以0给去掉了)[B,L,768]
        t = token_embeddings * input_mask_expanded
        #[B,768]
        sum_embeddings = torch.sum(t, 1)

        # [B,768],最大值为seq_len
        sum_mask = input_mask_expanded.sum(1)
        #限定每个元素的最小值是1e-9，保证分母不为0
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        #得到最后的具体embedding的每一个维度的值——元素相除
        output_vectors.append(sum_embeddings / sum_mask)

        #列拼接
        output_vector = torch.cat(output_vectors, 1)

        return  output_vector

    def encoding(self,inputs):
        self.bert.eval()
        with torch.no_grad():
            output = self.bert(**inputs, return_dict=True, output_hidden_states=True)
            embedding = output.hidden_states[-1]
            embedding = self.pooling(embedding, inputs)
        return embedding


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'MutilLabelClassification': (BertConfig, BertModel, BertTokenizer),
}


# task_c
class ColaDataset(DatasetBase):

    def __init__(self,
                 data_name,
                 data_dir,
                 data_type,
                 process_piplines: List[Callable],
                 **kwargs):
        super().__init__(data_name, data_dir, data_type, process_piplines, **kwargs)

    @classmethod
    def get_labels(self) -> List[str]:
        return [
            '1.1 threats of harm',
            '1.2 incitement and encouragement of harm',
            '2.1 descriptive attacks',
            '2.2 aggressive and emotive attacks',
            '2.3 dehumanising attacks & overt sexual objectification',
            '3.1 casual use of gendered slurs, profanities, and insults',
            '3.2 immutable gender differences and gender stereotypes',
            '3.3 backhanded gendered compliments',
            '3.4 condescending explanations or unwelcome advice',
            '4.1 supporting mistreatment of individual women',
            '4.2 supporting systemic discrimination against women as a group',
            ]

    def read_data(self, input_file: str) -> Any:
        with open(input_file, 'r', encoding='utf-8') as f:
            return list(f.readlines())

    def create_examples(self, data: Any, data_type: str, **kwargs) -> List[Dict[str, Any]]:
        test_mode = data_type == "test"
        examples = []
        for (i, line_old) in enumerate(data):
            line = json.loads(line_old)
            guid = f"{data_type}-{i}"
            text = line['text']
            label = None if test_mode else line['label_c']
            # if label == None:
            #     print(test_mode)
            #     print(line['label_c'])
            #     assert False
            if label == 'none':
                continue
            examples.append(dict(guid=guid, text=text, label=label))
        if data_type == 'test':
            pass
        else:
            random.shuffle(examples)
        return examples


class ProcessEncodeText(ProcessBase):
    """ 编码单句任务文本，在原有example上追加 """

    def __init__(self, tokenizer, tokenizer_params, return_input_length=False):
        self.tokenizer = tokenizer
        self.tokenizer_params = tokenizer_params
        self.return_input_length = return_input_length

    def __call__(self, example):
        inputs = self.tokenizer(example["text"], **self.tokenizer_params)
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        if self.return_input_length:
            inputs["input_length"] = inputs["attention_mask"].sum().item()
        example = dict(example, **inputs)
        return example


class ProcessEncodeLabel(ProcessBase):
    """ 编码单标签文本标签 """

    def __init__(self, label2id):
        self.label2id = label2id

    def __call__(self, example):
        example["label"] = self.label2id.get(example["label"], None)
        return example


def load_data(data_name, data_dir, data_type, tokenizer, max_sequence_length):
    process_piplines = [ProcessEncodeText(tokenizer,
                                          tokenizer_params={
                                              "padding": "max_length",
                                              "truncation": "longest_first",
                                              "max_length": max_sequence_length,
                                              "return_tensors": "pt",
                                          }),
                        ProcessEncodeLabel(ColaDataset.label2id())
                        ]
    return ColaDataset(data_name=data_name,
                       data_dir=data_dir,
                       data_type=data_type,
                       process_piplines=process_piplines
                       )


def main():
    opts = Argparser().get_training_arguments()
    logger = Logger(opts=opts)
    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]
    # data processor
    logger.info("initializing data processor")
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    train_dataset = load_data(opts.train_input_file, opts.data_dir, "train", tokenizer, opts.train_max_seq_length)
    for index in range(len(train_dataset)):
        if train_dataset[index]["label"] == None:
            print(train_dataset[index])
    dev_dataset = load_data(opts.eval_input_file, opts.data_dir, "dev", tokenizer, opts.eval_max_seq_length)
    opts.num_labels = train_dataset.num_labels
    # model
    logger.info("initializing model and config")
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    model = model_class.from_pretrained(opts.pretrained_model_path, config=config)
    model.to(opts.device)
    # trainer
    logger.info("initializing traniner")
    trainer = TextClassifierTrainer(opts=opts,
                                    model=model,
                                    metrics=[F1(num_classes=opts.num_labels)],
                                    logger=logger
                                    )
    # do train
    if opts.do_train:
        trainer.train(train_data=train_dataset, dev_data=dev_dataset, state_to_save={'vocab': tokenizer})
    if opts.do_eval:
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        if opts.eval_all_checkpoints:
            checkpoints = find_all_checkpoints(checkpoint_dir=opts.output_dir)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.evaluate(dev_data=dev_dataset, save_result=True, save_dir=prefix)

    if opts.do_predict:
        test_dataset = load_data(opts.test_input_file, opts.data_dir, "test", tokenizer, opts.test_max_seq_length)
        checkpoints = []
        if opts.checkpoint_predict_code is not None:
            checkpoint = os.path.join(opts.output_dir, opts.checkpoint_predict_code)
            check_dir(checkpoint)
            checkpoints.append(checkpoint)
        if opts.eval_all_checkpoints:
            checkpoints = find_all_checkpoints(checkpoint_dir=opts.output_dir)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split("/")[-1]
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(opts.device)
            trainer.model = model
            trainer.predict(test_data=test_dataset, save_result=True, save_dir=prefix)


if __name__ == "__main__":
    main()
