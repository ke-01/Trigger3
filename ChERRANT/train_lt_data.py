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
tp_diff = list(set(tp_pos) - set(tp_pos_llm))

# fp
fp_pos=load_list_from_file('train_trigger/qq_train_trigger_gector_fp_pos.json')
fp_pos=list(set(fp_pos))
fp_pos_llm=load_list_from_file('train_trigger/qq_train_trigger_qwen_gector_fp_pos.json')
fp_pos_llm=list(set(fp_pos_llm))
fp_diff = list(set(fp_pos) - set(fp_pos_llm))

# fn
fn_pos=load_list_from_file('train_trigger/qq_train_trigger_gector_fn_pos.json')
fn_pos=list(set(fn_pos))
fn_pos_llm=load_list_from_file('train_trigger/qq_train_trigger_qwen_gector_fn_pos.json')
fn_pos_llm=list(set(fn_pos_llm))
fn_diff = list(set(fn_pos) - set(fn_pos_llm))

# or
intersection_set = set(fn_diff) | set(fp_diff) |set(tp_diff)
diff_3 = list(intersection_set)

# src file
train_part_src='qq_train_part.txt'
with open(train_part_src,'r',encoding='utf8')as fr:
    train_src = fr.readlines()
    

train_part_s2e='qq_train_part_gector.txt'
with open(train_part_s2e,'r',encoding='utf8')as fr:
    train_s2e = fr.readlines()
    

unique_values = find_unique_values(len(train_src),diff_3)

# for lt
pos_lines_src = [train_src[value] for value in diff_3[:5000]]
pos_lines_trg = [train_s2e[value] for value in diff_3[:5000]]

neg_s=unique_values[:len(diff_3[:5000])]
neg_lines_src = [train_src[value] for value in neg_s]
neg_lines_trg = [train_s2e[value] for value in neg_s]

src_all=pos_lines_src+neg_lines_src
trg_all=pos_lines_trg+neg_lines_trg


lt_train_src_file='qq_train_lt_gector_src.txt'
with open(lt_train_src_file, 'w', encoding='utf-8') as f:
    for line in src_all:
        f.write(line) 
        
        
lt_train_trg_file='qq_train_lt_gector_trg.txt'
with open(lt_train_trg_file, 'w', encoding='utf-8') as f:
    for line in trg_all:
        f.write(line) 