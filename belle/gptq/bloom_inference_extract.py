import time

import torch
import torch.nn as nn
import json
from tqdm import tqdm
from gptq import *
from modelutils import *
from quant import *

from transformers import AutoTokenizer

DEV = torch.device('cuda:0')

def get_bloom(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

def load_quant(model, checkpoint, wbits, groupsize):
    from transformers import BloomConfig, BloomForCausalLM
    config = BloomConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = BloomForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize)

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )

    parser.add_argument(
        '--text', type=str,
        help='hello'
    )

    parser.add_argument(
        '--min_length', type=int, default=1,
        help='The minimum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--max_length', type=int, default=512,
        help='The maximum length of the sequence to be generated.'
    )

    parser.add_argument(
        '--top_p', type=float , default=0.95,
        help='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
    )

    parser.add_argument(
        '--temperature', type=float, default=0.8,
        help='The value used to module the next token probabilities.'
    )
    parser.add_argument(
        '--batchsize', type=int, default=32,
    )

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_bloom(args.model)
        model.eval()

    model.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"

    # with open('../new/train.json', 'r', encoding='utf-8') as read_file, open('../new/train_summary.json', 'w',
    #                                                                        encoding='utf-8') as write_file:
    #     read_lines = read_file.readlines()
    #     total_number = len(read_lines)
    #     for index in tqdm(range(total_number)):
    #         line = read_lines[index]
    #         total_message = json.loads(line)
    #         content = total_message['content']
    #         title = total_message['title']
    #         url = total_message['url']
    #         pub_time = total_message['pub_time']
    #         entities = total_message['entities']
    #
    #         content_list = []
    #         while len(content) >= 20:
    #             content_list.append(content[:1024])
    #             content = content[1024:]
    #         restext_list = []
    #         for content_index, content_item in enumerate(content_list):
    #             response = '下面是一篇文章的一部分，请你帮我将他缩写为长度在96字左右的摘要。内容如下：' + content_item
    #             inputs = 'Human: ' + response.strip() + '\n\nAssistant:'
    #             input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)
    #             with torch.no_grad():
    #                 generated_ids = model.generate(
    #                     input_ids,
    #                     do_sample=True,
    #                     min_length=args.min_length,
    #                     max_length=args.max_length,
    #                     top_p=args.top_p,
    #                     temperature=args.temperature,
    #                 )
    #
    #             restext = tokenizer.decode([el.item() for el in generated_ids[0]])
    #             restext_list.append(restext.replace(inputs, ''))
    #         write_file.write(
    #             json.dumps(
    #                 {"pub_time": pub_time, "title": title, "id": index, "url": url, 'content': ' '.join(restext_list),
    #                  "entities": entities}, ensure_ascii=False) + '\n')
    # print('train done.')
    #
    # with open('../new/eval.json', 'r', encoding='utf-8') as read_file, open('../new/eval_summary.json', 'w',
    #                                                                        encoding='utf-8') as write_file:
    #     read_lines = read_file.readlines()
    #     total_number = len(read_lines)
    #     for index in tqdm(range(total_number)):
    #         line = read_lines[index]
    #         total_message = json.loads(line)
    #         content = total_message['content']
    #         title = total_message['title']
    #         url = total_message['url']
    #         pub_time = total_message['pub_time']
    #         entities = total_message['entities']
    #
    #         content_list = []
    #         while len(content) >= 20:
    #             content_list.append(content[:1024])
    #             content = content[1024:]
    #         restext_list = []
    #         for content_index, content_item in enumerate(content_list):
    #             response = '下面是一篇文章的一部分，请你帮我将他缩写为长度在96字左右的摘要。内容如下：' + content_item
    #             inputs = 'Human: ' + response.strip() + '\n\nAssistant:'
    #             input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)
    #             with torch.no_grad():
    #                 generated_ids = model.generate(
    #                     input_ids,
    #                     do_sample=True,
    #                     min_length=args.min_length,
    #                     max_length=args.max_length,
    #                     top_p=args.top_p,
    #                     temperature=args.temperature,
    #                 )
    #
    #             restext = tokenizer.decode([el.item() for el in generated_ids[0]])
    #             restext_list.append(restext.replace(inputs, ''))
    #         write_file.write(
    #             json.dumps(
    #                 {"pub_time": pub_time, "title": title, "id": index, "url": url, 'content': ' '.join(restext_list),
    #                  "entities": entities}, ensure_ascii=False) + '\n')
    # print('eval done.')
    #
    # with open('../new/test.json', 'r', encoding='utf-8') as read_file, open('../new/test_summary.json', 'w',
    #                                                                        encoding='utf-8') as write_file:
    #     read_lines = read_file.readlines()
    #     total_number = len(read_lines)
    #     for index in tqdm(range(total_number)):
    #         line = read_lines[index]
    #         total_message = json.loads(line)
    #         content = total_message['content']
    #         title = total_message['title']
    #         url = total_message['url']
    #         pub_time = total_message['pub_time']
    #         entities = total_message['entities']
    #
    #         content_list = []
    #         while len(content) >= 20:
    #             content_list.append(content[:1024])
    #             content = content[1024:]
    #         restext_list = []
    #         for content_index, content_item in enumerate(content_list):
    #             response = '下面是一篇文章的一部分，请你帮我将他缩写为长度在96字左右的摘要。内容如下：' + content_item
    #             inputs = 'Human: ' + response.strip() + '\n\nAssistant:'
    #             input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)
    #             with torch.no_grad():
    #                 generated_ids = model.generate(
    #                     input_ids,
    #                     do_sample=True,
    #                     min_length=args.min_length,
    #                     max_length=args.max_length,
    #                     top_p=args.top_p,
    #                     temperature=args.temperature,
    #                 )
    #
    #             restext = tokenizer.decode([el.item() for el in generated_ids[0]])
    #             restext_list.append(restext.replace(inputs, ''))
    #         write_file.write(
    #             json.dumps(
    #                 {"pub_time": pub_time, "title": title, "id": index, "url": url, 'content': ' '.join(restext_list),
    #                  "entities": entities}, ensure_ascii=False) + '\n')
    # print('test done.')
    #
    # print('all done.')
    # with open('../new/train.json', 'r', encoding='utf-8') as read_file, open('../new/train_extract.json', 'w',
    #                                                                        encoding='utf-8') as write_file:
    #     read_lines = read_file.readlines()
    #     total_number = len(read_lines)
    #     for index in tqdm(range(total_number)):
    #         line = read_lines[index]
    #         total_message = json.loads(line)
    #         content = total_message['content']
    #         title = total_message['title']
    #         url = total_message['url']
    #         pub_time = total_message['pub_time']
    #         entities = total_message['entities']
    #
    #         content_list = []
    #         index = 192
    #         content_part = content[:index]
    #         while len(content) >= (index+1):
    #             while len(content) >= (index+1) and (content[index] != '。' and content[index] != '？' and content[index] != '?' and content[index] != '!' and content[index] != '！'):
    #                 content_part += content[index]
    #                 index += 1
    #             if len(content) >= (index+1):
    #                 content_part += content[index]
    #                 index += 1
    #                 content_list.append(content_part)
    #                 content_part = content[index:index+192]
    #                 index += 192
    #             else:
    #                 content_list.append(content_part)
    #         restext_list = []
    #         for content_index, content_item in enumerate(content_list):
    #             response = '对于关系与事件，我们通常以如下方式表示：小明-父亲-大明， 王二-朋友-张三， 小明-掌管-董事会， 周杰伦-举办-演唱会。下面是一篇文章的一部分，请你帮我将其中的实体、实体间的关系、发生的事件以及关键词抽取出来，关系与事件用前面所说的通常的方式表示，如果有多个，请用“,”隔开。内容如下：' + content_item
    #             inputs = 'Human: ' + response.strip() + '\n\nAssistant:'
    #             input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)
    #             with torch.no_grad():
    #                 generated_ids = model.generate(
    #                     input_ids,
    #                     do_sample=True,
    #                     min_length=args.min_length,
    #                     max_length=args.max_length,
    #                     top_p=args.top_p,
    #                     temperature=args.temperature,
    #                 )
    #
    #             restext = tokenizer.decode([el.item() for el in generated_ids[0]])
    #             restext_list.append(restext.replace(inputs, ''))
    #         write_file.write(
    #             json.dumps(
    #                 {"pub_time": pub_time, "title": title, "id": index, "url": url, 'content': ','.join(restext_list),
    #                  "entities": entities}, ensure_ascii=False) + '\n')
    # print('train done.')
    #
    # with open('../new/eval.json', 'r', encoding='utf-8') as read_file, open('../new/eval_extract.json', 'w',
    #                                                                        encoding='utf-8') as write_file:
    #     read_lines = read_file.readlines()
    #     total_number = len(read_lines)
    #     for index in tqdm(range(total_number)):
    #         line = read_lines[index]
    #         total_message = json.loads(line)
    #         content = total_message['content']
    #         title = total_message['title']
    #         url = total_message['url']
    #         pub_time = total_message['pub_time']
    #         entities = total_message['entities']
    #
    #         content_list = []
    #         index = 192
    #         content_part = content[:index]
    #         while len(content) >= (index + 1):
    #             while len(content) >= (index + 1) and (
    #                     content[index] != '。' and content[index] != '？' and content[index] != '?' and content[
    #                 index] != '!' and content[index] != '！'):
    #                 content_part += content[index]
    #                 index += 1
    #             if len(content) >= (index + 1):
    #                 content_part += content[index]
    #                 index += 1
    #                 content_list.append(content_part)
    #                 content_part = content[index:index + 192]
    #                 index += 192
    #             else:
    #                 content_list.append(content_part)
    #         restext_list = []
    #         for content_index, content_item in enumerate(content_list):
    #             response = '对于关系与事件，我们通常以如下方式表示：小明-父亲-大明， 王二-朋友-张三， 小明-掌管-董事会， 周杰伦-举办-演唱会。下面是一篇文章的一部分，请你帮我将其中的实体、实体间的关系、发生的事件以及关键词抽取出来，关系与事件用前面所说的通常的方式表示，如果有多个，请用“,”隔开。内容如下：' + content_item
    #             inputs = 'Human: ' + response.strip() + '\n\nAssistant:'
    #             input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)
    #             with torch.no_grad():
    #                 generated_ids = model.generate(
    #                     input_ids,
    #                     do_sample=True,
    #                     min_length=args.min_length,
    #                     max_length=args.max_length,
    #                     top_p=args.top_p,
    #                     temperature=args.temperature,
    #                 )
    #
    #             restext = tokenizer.decode([el.item() for el in generated_ids[0]])
    #             restext_list.append(restext.replace(inputs, ''))
    #         write_file.write(
    #             json.dumps(
    #                 {"pub_time": pub_time, "title": title, "id": index, "url": url, 'content': ','.join(restext_list),
    #                  "entities": entities}, ensure_ascii=False) + '\n')
    # print('eval done.')
    #
    # with open('../new/test.json', 'r', encoding='utf-8') as read_file, open('../new/test_extract.json', 'w',
    #                                                                        encoding='utf-8') as write_file:
    #     read_lines = read_file.readlines()
    #     total_number = len(read_lines)
    #     for index in tqdm(range(total_number)):
    #         line = read_lines[index]
    #         total_message = json.loads(line)
    #         content = total_message['content']
    #         title = total_message['title']
    #         url = total_message['url']
    #         pub_time = total_message['pub_time']
    #         entities = total_message['entities']
    #
    #         content_list = []
    #         index = 192
    #         content_part = content[:index]
    #         while len(content) >= (index + 1):
    #             while len(content) >= (index + 1) and (
    #                     content[index] != '。' and content[index] != '？' and content[index] != '?' and content[
    #                 index] != '!' and content[index] != '！'):
    #                 content_part += content[index]
    #                 index += 1
    #             if len(content) >= (index + 1):
    #                 content_part += content[index]
    #                 index += 1
    #                 content_list.append(content_part)
    #                 content_part = content[index:index + 192]
    #                 index += 192
    #             else:
    #                 content_list.append(content_part)
    #         restext_list = []
    #         for content_index, content_item in enumerate(content_list):
    #             response = '对于关系与事件，我们通常以如下方式表示：小明-父亲-大明， 王二-朋友-张三， 小明-掌管-董事会， 周杰伦-举办-演唱会。下面是一篇文章的一部分，请你帮我将其中的实体、实体间的关系、发生的事件以及关键词抽取出来，关系与事件用前面所说的通常的方式表示，如果有多个，请用“,”隔开。内容如下：' + content_item
    #             inputs = 'Human: ' + response.strip() + '\n\nAssistant:'
    #             input_ids = tokenizer.encode(inputs, return_tensors="pt").to(DEV)
    #             with torch.no_grad():
    #                 generated_ids = model.generate(
    #                     input_ids,
    #                     do_sample=True,
    #                     min_length=args.min_length,
    #                     max_length=args.max_length,
    #                     top_p=args.top_p,
    #                     temperature=args.temperature,
    #                 )
    #
    #             restext = tokenizer.decode([el.item() for el in generated_ids[0]])
    #             restext_list.append(restext.replace(inputs, ''))
    #         write_file.write(
    #             json.dumps(
    #                 {"pub_time": pub_time, "title": title, "id": index, "url": url, 'content': ','.join(restext_list),
    #                  "entities": entities}, ensure_ascii=False) + '\n')
    # print('test done.')
    #
    # print('all done.')


    batch_size = args.batchsize

    dev_data = []
    with open('../new/train.json', 'r', encoding='utf-8') as read_file:
        read_lines = read_file.readlines()
        total_number = len(read_lines)
        for index in tqdm(range(total_number)):
            line = read_lines[index]
            total_message = json.loads(line)
            content = total_message['content']
            content_list = []
            index = 192
            content_part = content[:index]
            while len(content) >= (index + 1):
                while len(content) >= (index + 1) and (
                        content[index] != '。' and content[index] != '？' and content[index] != '?' and content[
                    index] != '!' and content[index] != '！'):
                    content_part += content[index]
                    index += 1
                if len(content) >= (index + 1):
                    content_part += content[index]
                    index += 1
                    content_list.append(content_part)
                    content_part = content[index:index + 192]
                    index += 192
                else:
                    content_list.append(content_part)
            content_list_len = len(content_list)
            if content_list_len >= 32:
                content_list = content_list[:32]
            else:
                content_list.extend(['<pad>' * 96] * (32 - content_list_len))
            assert len(content_list) == 32
            dev_data.extend(content_list)

    res = []
    skip_special_tokens = True
    clean_up_tokenization_spaces = True
    for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data), unit="batch"):
        batch = dev_data[i:i + batch_size]
        batch_text = []
        for item in batch:
            response = '对于关系与事件，我们通常以如下方式表示：小明-父亲-大明， 王二-朋友-张三， 小明-掌管-董事会， 周杰伦-举办-演唱会。下面是一篇文章的一部分，请你帮我将其中的实体、实体间的关系、发生的事件以及关键词抽取出来，关系与事件用前面所说的通常的方式表示，如果有多个，请用“,”隔开。内容如下：' + \
                       item
            input_text = "Human: " + response + "\n\nAssistant: "
            batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text)

        with torch.autocast("cuda"):
            features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True,
                                 max_length=args.max_length)
            input_ids = features['input_ids'].to("cuda")
            attention_mask = features['attention_mask'].to("cuda")

            output_texts = model.generate(
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                # num_beams=4,
                # do_sample=False,
                # min_new_tokens=1,
                # max_new_tokens=512,
                # early_stopping=True,
                input_ids,
                attention_mask=attention_mask,
                # do_sample=True,
                min_length=args.min_length,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        output_texts = tokenizer.batch_decode(
            output_texts.cpu().numpy().tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        for i in range(len(output_texts)):
            input_text = batch_text[i]
            input_text = input_text.replace(tokenizer.bos_token, "")
            predict_text = output_texts[i][len(input_text):]
            res.append({"input": input_text, "predict": predict_text})

    with open('../new/train_extract.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print('train done.')

    dev_data = []
    with open('../new/eval.json', 'r', encoding='utf-8') as read_file:
        read_lines = read_file.readlines()
        total_number = len(read_lines)
        for index in tqdm(range(total_number)):
            line = read_lines[index]
            total_message = json.loads(line)
            content = total_message['content']
            content_list = []
            index = 192
            content_part = content[:index]
            while len(content) >= (index + 1):
                while len(content) >= (index + 1) and (
                        content[index] != '。' and content[index] != '？' and content[index] != '?' and content[
                    index] != '!' and content[index] != '！'):
                    content_part += content[index]
                    index += 1
                if len(content) >= (index + 1):
                    content_part += content[index]
                    index += 1
                    content_list.append(content_part)
                    content_part = content[index:index + 192]
                    index += 192
                else:
                    content_list.append(content_part)
            content_list_len = len(content_list)
            if content_list_len >= 32:
                content_list = content_list[:32]
            else:
                content_list.extend(['<pad>' * 96] * (32 - content_list_len))
            assert len(content_list) == 32
            dev_data.extend(content_list)

    res = []
    skip_special_tokens = True
    clean_up_tokenization_spaces = True
    for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data), unit="batch"):
        batch = dev_data[i:i + batch_size]
        batch_text = []
        for item in batch:
            response = '对于关系与事件，我们通常以如下方式表示：小明-父亲-大明， 王二-朋友-张三， 小明-掌管-董事会， 周杰伦-举办-演唱会。下面是一篇文章的一部分，请你帮我将其中的实体、实体间的关系、发生的事件以及关键词抽取出来，关系与事件用前面所说的通常的方式表示，如果有多个，请用“,”隔开。内容如下：' + \
                       item
            input_text = "Human: " + response + "\n\nAssistant: "
            batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text)

        with torch.autocast("cuda"):
            features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True,
                                 max_length=args.max_length)
            input_ids = features['input_ids'].to("cuda")
            attention_mask = features['attention_mask'].to("cuda")

            output_texts = model.generate(
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                # num_beams=4,
                # do_sample=False,
                # min_new_tokens=1,
                # max_new_tokens=512,
                # early_stopping=True,
                input_ids,
                attention_mask=attention_mask,
                # do_sample=True,
                min_length=args.min_length,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        output_texts = tokenizer.batch_decode(
            output_texts.cpu().numpy().tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        for i in range(len(output_texts)):
            input_text = batch_text[i]
            input_text = input_text.replace(tokenizer.bos_token, "")
            predict_text = output_texts[i][len(input_text):]
            res.append({"input": input_text, "predict": predict_text})

    with open('../new/eval_extract.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print('eval done.')

    dev_data = []
    with open('../new/test.json', 'r', encoding='utf-8') as read_file:
        read_lines = read_file.readlines()
        total_number = len(read_lines)
        for index in tqdm(range(total_number)):
            line = read_lines[index]
            total_message = json.loads(line)
            content = total_message['content']
            content_list = []
            index = 192
            content_part = content[:index]
            while len(content) >= (index + 1):
                while len(content) >= (index + 1) and (
                        content[index] != '。' and content[index] != '？' and content[index] != '?' and content[
                    index] != '!' and content[index] != '！'):
                    content_part += content[index]
                    index += 1
                if len(content) >= (index + 1):
                    content_part += content[index]
                    index += 1
                    content_list.append(content_part)
                    content_part = content[index:index + 192]
                    index += 192
                else:
                    content_list.append(content_part)
            content_list_len = len(content_list)
            if content_list_len >= 32:
                content_list = content_list[:32]
            else:
                content_list.extend(['<pad>' * 96] * (32 - content_list_len))
            assert len(content_list) == 32
            dev_data.extend(content_list)

    res = []
    skip_special_tokens = True
    clean_up_tokenization_spaces = True
    for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data), unit="batch"):
        batch = dev_data[i:i + batch_size]
        batch_text = []
        for item in batch:
            response = '对于关系与事件，我们通常以如下方式表示：小明-父亲-大明， 王二-朋友-张三， 小明-掌管-董事会， 周杰伦-举办-演唱会。下面是一篇文章的一部分，请你帮我将其中的实体、实体间的关系、发生的事件以及关键词抽取出来，关系与事件用前面所说的通常的方式表示，如果有多个，请用“,”隔开。内容如下：' + \
                       item
            input_text = "Human: " + response + "\n\nAssistant: "
            batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text)

        with torch.autocast("cuda"):
            features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True,
                                 max_length=args.max_length)
            input_ids = features['input_ids'].to("cuda")
            attention_mask = features['attention_mask'].to("cuda")

            output_texts = model.generate(
                # input_ids=input_ids,
                # attention_mask=attention_mask,
                # num_beams=4,
                # do_sample=False,
                # min_new_tokens=1,
                # max_new_tokens=512,
                # early_stopping=True,
                input_ids,
                attention_mask=attention_mask,
                # do_sample=True,
                min_length=args.min_length,
                max_length=args.max_length,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        output_texts = tokenizer.batch_decode(
            output_texts.cpu().numpy().tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        for i in range(len(output_texts)):
            input_text = batch_text[i]
            input_text = input_text.replace(tokenizer.bos_token, "")
            predict_text = output_texts[i][len(input_text):]
            res.append({"input": input_text, "predict": predict_text})

    with open('../new/test_extract.json', 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print('test done.')

    print('all done.')