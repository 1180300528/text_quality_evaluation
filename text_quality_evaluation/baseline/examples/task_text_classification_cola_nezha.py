import os
import csv
import json
import sys
import random
import torch
import torch.nn as nn
sys.path.append('..')
from torch.utils.data import Dataset, DataLoader
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
from transformers import RoFormerTokenizer, RoFormerForSequenceClassification, RoFormerConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from model import HeteroGNN_with_RoBERTa
from NEZHA.configuration_nezha import NeZhaConfig
from create_graph import MyOwnDataset


maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'mine': (BertConfig, HeteroGNN_with_RoBERTa, BertTokenizer),
    'roformer': (RoFormerConfig, RoFormerForSequenceClassification, RoFormerTokenizer),
    'deberta': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
}


def main():
    opts = Argparser().get_training_arguments()
    logger = Logger(opts=opts)
    # device
    logger.info("initializing device")
    opts.device, opts.device_num = prepare_device(opts.device_id)
    seed_everything(opts.seed)
    logger.info("initializing model and config")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[opts.model_type]
    tokenizer = tokenizer_class.from_pretrained(opts.pretrained_model_path, do_lower_case=opts.do_lower_case)
    config = config_class.from_pretrained(opts.pretrained_model_path, num_labels=opts.num_labels)
    model = HeteroGNN_with_RoBERTa(opts.pretrained_model_path, config, tokenizer, opts.device)
    model.to(opts.device)
    # # 扩充词表
    # with open(os.path.join(opts.data_dir, 'roberta_single_subtraction.csv'), 'r', encoding='utf-8') as f:
    #     word_count = list(csv.reader(f))
    # word_count = word_count[1:3002]
    # new_tokens = [item[0] for item in word_count]
    # num_added_toks = tokenizer.add_tokens(new_tokens)
    # model.model.resize_token_embeddings(len(tokenizer))
    # model.model4tag.resize_token_embeddings(len(tokenizer))
    # tokenizer.save_pretrained(opts.pretrained_cache_dir)

    # data processor
    logger.info("initializing data processor")
    train_dataset = MyOwnDataset('./train_dataset')
    dev_dataset = MyOwnDataset('./eval_dataset')
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
            model = HeteroGNN_with_RoBERTa(opts.pretrained_model_path, config, tokenizer, opts.device)
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
            model.to(opts.device)
            # model.model.resize_token_embeddings(len(tokenizer))
            # model.model4tag.resize_token_embeddings(len(tokenizer))
            trainer.model = model
            trainer.evaluate(dev_data=dev_dataset, save_result=True, save_dir=prefix)

    if opts.do_predict:
        test_dataset = MyOwnDataset('./test_dataset')
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
            model = HeteroGNN_with_RoBERTa(opts.pretrained_model_path, config, tokenizer, opts.device)
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))
            model.to(opts.device)
            # model.model.resize_token_embeddings(len(tokenizer))
            # model.model4tag.resize_token_embeddings(len(tokenizer))
            trainer.model = model
            trainer.predict(test_data=test_dataset, save_result=True, save_dir=prefix)


if __name__ == "__main__":
    main()
