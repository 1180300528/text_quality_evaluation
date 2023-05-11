# -*- coding: utf-8 -*-
import json
import time
import openai
openai.api_key = "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"

with open('../data/new/train.json', 'r', encoding='utf-8') as read_file, open('../data/new/train_summary.json', 'w', encoding='utf-8') as write_file:
    read_lines = read_file.readlines()
    total_number = len(read_lines)
    index = 0
    while index < total_number:
        try:
            line = read_lines[index]
            total_message = json.loads(line)
            content = total_message['content']
            title = total_message['title']
            url = total_message['url']
            pub_time = total_message['pub_time']
            entities = total_message['entities']

            content_list = []
            while len(content) >= 20:
                content_list.append(content[:1024])
                content = content[1024:]
            restext_list = []
            for content_index, content_item in enumerate(content_list):
                # https://zhuanlan.zhihu.com/p/606573556
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    # max_tokens=112,
                    temperature=0.0,
                    # top_p=0.1,  # 与temperature二选一
                    # frequency_penalty=0.0,
                    # presence_penalty=0.0,     # 两个penalty取值为-2.0~2.0, 越小重复程度越高， 主题越紧密
                    # stop=['', '', '', '', ],      # 停止字符, 最大长度为4的字符串列表，一旦生成的tokens包含其中的内容，将停止生成并返回结果
                    messages=[
                        {"role": "system", "content": "现在你是一个合格的文章摘要者"},
                        {"role": "system",
                         "content": "下面是一篇题目为《" + title + "》的文章的一部分， 请你帮我将他缩写为长度在112字左右的摘要。"},
                        {"role": "user", "content": content_item},
                        # {"role": "assistant", "content": content_reply},
                        # {"role": "system", "content": ""},
                        # {"role": "user", "content": ""},
                    ]
                )
                restext = response.choices[0].message.content
                restext_list.append(restext)
            write_file.write(
                json.dumps({"pub_time": pub_time, "title": title, "id": index, "url": url, 'content': ' '.join(restext_list), "entities": entities}, ensure_ascii=False) + '\n')
        except Exception as e:
            time.sleep(30)
            continue

        index += 1
