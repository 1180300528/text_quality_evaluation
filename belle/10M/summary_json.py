import json
from tqdm import tqdm
with open('./FewCLUE_CSL.json', 'r', encoding='utf-8') as read_file_1, \
        open('./FewCLUE_TNEWS.json', 'r', encoding='utf-8') as read_file_2, \
        open('./FinRE.json', 'r', encoding='utf-8') as read_file_3, \
        open('./Doc2EDAG.json', 'r', encoding='utf-8') as read_file_4, \
        open('./train_for_extraction.json', 'w', encoding='utf-8') as write_file:
    read_lines = read_file_1.readlines()
    read_lines.extend(read_file_2.readlines())
    read_lines.extend(read_file_3.readlines())
    read_lines.extend(read_file_4.readlines())
    for line in tqdm(read_lines):
        line = json.loads(line)
        write_file.write(json.dumps(line, ensure_ascii=False) + '\n')
