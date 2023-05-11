import sys, os
import argparse
import json
import torch
from peft import PeftModel
import transformers
import gradio as gr
from tqdm import tqdm

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def get_model(base_model):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
            )

    return model


def load_dev_data(dev_file_path = '../new/train/json'):
    dev_data = []
    with open(dev_file_path, 'r', encoding='utf-8') as read_file:
        read_lines = read_file.readlines()
        total_number = len(read_lines)
        for index in tqdm(range(total_number)):
            line = read_lines[index]
            total_message = json.loads(line)
            content = total_message['content']
            content_list = []
            index = 192
            content_part = content[:index]
            while len(content) >= (index+1):
                while len(content) >= (index+1) and (content[index] != '。' and content[index] != '？' and content[index] != '?' and content[index] != '!' and content[index] != '！'):
                    content_part += content[index]
                    index += 1
                if len(content) >= (index+1):
                    content_part += content[index]
                    index += 1
                    content_list.append(content_part)
                    content_part = content[index:index+192]
                    index += 192
                else:
                    content_list.append(content_part)
            content_list_len = len(content_list)
            if content_list_len >= 32:
                content_list = content_list[:32]
            else:
                content_list.extend(['<pad>' * 96] * ( 32-content_list_len))
            assert len(content_list) == 32
            dev_data.extend(content_list)
    return dev_data

def generate_text(dev_data, batch_size, tokenizer, model, skip_special_tokens = True, clean_up_tokenization_spaces=True):
    res = []
    for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data), unit="batch"):
        batch = dev_data[i:i+batch_size]
        batch_text = []
        for item in batch:
            response = '对于关系与事件，我们通常以如下方式表示：小明-父亲-大明， 王二-朋友-张三， 小明-掌管-董事会， 周杰伦-举办-演唱会。下面是一篇文章的一部分，请你帮我将其中的实体、实体间的关系、发生的事件以及关键词抽取出来，关系与事件用前面所说的通常的方式表示，如果有多个，请用“,”隔开。内容如下：' + item
            input_text = "Human: " + response + "\n\nAssistant: "
            batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text)

        with torch.autocast("cuda"):
            features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True, max_length = args.inputs_max_length)
            input_ids = features['input_ids'].to("cuda")
            attention_mask = features['attention_mask'].to("cuda")

            output_texts = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams = 2,
                do_sample = False,
                min_new_tokens=1,
                max_new_tokens=args.inputs_max_length,
                early_stopping= True,
                # input_ids,
                # do_sample=True,
                # min_length=args.min_length,
                # max_length=args.max_length,
                # top_p=args.top_p,
                # temperature=args.temperature,
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
            res.append({"input":input_text,"predict":predict_text})
    return res


def main(args):
    dev_data = load_dev_data(args.dev_file)[:args.datasets_max_length] if args.datasets_max_length!=-1 else load_dev_data(args.dev_file)
    res = generate_text(dev_data, batch_size, tokenizer, model)
    with open(args.output_file, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="pretrained language model")
    parser.add_argument("--datasets_max_length", type=int, default=96, help="max length of dataset")
    parser.add_argument("--inputs_max_length", type=int, default=96, help="max length of inputs")
    parser.add_argument("--dev_batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--lora_weights", default="", type=str, help="use lora")
    parser.add_argument("--output_file", type=str, default="data_dir/predictions.json")

    args = parser.parse_args()
    batch_size = args.dev_batch_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = get_model(args.model_name_or_path)
    model.cuda()
    main(args)
    print("done.")