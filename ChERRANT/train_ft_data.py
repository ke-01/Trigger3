import pandas as pd
import numpy as np
import json

def load_list_from_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_unique_values(max_len,loaded_list):
    set1 = set(loaded_list)
    unique_values = [i for i in range(max_len) if i not in set1]
    return unique_values

# 依据 qq_train_trigger_data.sh得到的文件
# tp
tp_pos=load_list_from_file('train_trigger/qq_train_trigger_gector_tp_pos.json')
tp_pos=list(set(tp_pos))
tp_pos_llm=load_list_from_file('train_trigger/qq_train_trigger_qwen_gector_tp_pos.json')
tp_pos_llm=list(set(tp_pos_llm))
tp_diff = list(set(tp_pos) & set(tp_pos_llm))


# src file
train_part_src='qq_train_part.txt'
with open(train_part_src,'r',encoding='utf8')as fr:
    train_src = fr.readlines()
    
train_part_s2e='qq_train_part_gector.txt'
with open(train_part_s2e,'r',encoding='utf8')as fr:
    train_s2e = fr.readlines()
    
train_part_llm='qq_qwen_train_part_gector.txt'
with open(train_part_llm,'r',encoding='utf8')as fr:
    train_llm = fr.readlines()
    

unique_values = find_unique_values(len(train_src),tp_diff)

# for lt
pos_lines_src = [train_src[value] for value in tp_diff[:5000]]
pos_lines_s2e = [train_s2e[value] for value in tp_diff[:5000]]
pos_lines_llm = [train_llm[value] for value in tp_diff[:5000]]
pos_lines_src=pos_lines_src
pos_lines_trg=pos_lines_s2e[:2500]+pos_lines_llm[2500:]

neg_s=unique_values[:len(pos_lines_trg)]
neg_lines_src = [train_src[value] for value in neg_s]
neg_lines_s2e = [train_s2e[value] for value in neg_s]
neg_lines_llm = [train_llm[value] for value in neg_s]
neg_lines_src=neg_lines_src
neg_lines_trg=neg_lines_s2e[:2500]+neg_lines_llm[2500:]

src_all=pos_lines_src+neg_lines_src
trg_all=pos_lines_trg+neg_lines_trg


ft_train_src_file='qq_train_ft_gector_src.txt'
with open(ft_train_src_file, 'w', encoding='utf-8') as f:
    for line in src_all:
        f.write(line) 
        
        
ft_train_trg_file='qq_train_ft_gector_trg.txt'
with open(ft_train_trg_file, 'w', encoding='utf-8') as f:
    for line in trg_all:
        f.write(line) 