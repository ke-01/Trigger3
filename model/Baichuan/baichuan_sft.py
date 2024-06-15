from transformers import AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn.functional as F
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

device = "cuda" # the device to load the model onto

import argparse
from argparse import Namespace

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--test_file', type=str, default="", help='test file')
parser.add_argument('--small_pre_file', type=str, default="", help='small_pre_file')
parser.add_argument('--output_file', type=str, default="", help='output_file')

args = parser.parse_args()


base_path="./Baichuan2-7B-Chat"
adapter_path = "../LLaMA-Factory/saves/lora/Baichuan2-7B-Chat/"


def get_model():
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16,trust_remote_code=True)
    
    model.generation_config = GenerationConfig.from_pretrained(base_path)
    
    peft_model = PeftModel.from_pretrained(model, adapter_path,torch_dtype=torch.bfloat16).eval()

    peft_model.cuda()
    
    return tokenizer, peft_model

def generate(model,tokenizer,prompt,pre_line):
    with torch.no_grad():

        messages = [
            {"role": "system", "content": "给定一个查询文本和初步改写，你的任务是输出它的正确形式。"},
            {"role": "user", "content": prompt+"初步改写为："+pre_line}
        ]
        response = model.chat(tokenizer, messages) 

        response=remove_empty_lines_and_newlines(response)
        
        return response

def remove_empty_lines_and_newlines(text):
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    text_without_empty_lines = '\n'.join(non_empty_lines)
    text_without_empty_lines_and_newlines = text_without_empty_lines.replace("\n", "")
    return text_without_empty_lines_and_newlines


tokenizer, model = get_model()


# test
with open(args.test_file,'r',encoding='utf8')as fr:
    lines = fr.readlines()
with open(args.small_pre_file,'r',encoding='utf8')as fr:
    s2s_pre_lines = fr.readlines()
with open(args.outout_file,'w',encoding='utf8') as fw:
    for line, pre_line in tqdm(zip(lines, s2s_pre_lines), total=len(lines)):
        line = line.strip()
        pre_line=pre_line.strip()
        res=[]
        for a in generate(model,tokenizer,line,pre_line): 
            res.append(a)
        ress=''.join(res)
        fw.write( ress + '\n')





