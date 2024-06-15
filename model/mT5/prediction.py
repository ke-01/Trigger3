import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BertTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm

import argparse
from argparse import Namespace
parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--save_file_name', type=str, default="", help='bert type')
parser.add_argument('--test_file', type=str, default="", help='bert type')
parser.add_argument('--output_file', type=str, default="", help='model type')

args = parser.parse_args()

device="cuda"
saved_model_path='../models/'+args.save_file_name
tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_path).to(device)

test_file=args.test_file
output_file=args.output_file

with open(test_file, 'r', encoding='utf-8') as file:
    test_lines = file.readlines()

def process_string(sentence):
    max_input_length = 256
    input_encodings = tokenizer(sentence, 
                                max_length=max_input_length, 
                                truncation=True, 
                                return_tensors="pt").to(device)
    if "token_type_ids" in input_encodings.keys():
        input_encodings.pop("token_type_ids")
    output = model.generate(**input_encodings, 
                            num_beams=5,
                            no_repeat_ngram_size=4,
                            do_sample=False, 
                            early_stopping=True,
                            min_length=5, 
                            max_length=64,
                            return_dict_in_generate=True,
                            output_scores=True)
    decoded_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    generation = decoded_output.strip()
    correction = generation.split("</s>")[0]

    return correction

with open(output_file, "w", encoding="utf-8") as f:
    for string in tqdm(test_lines, desc="Processing"):
        processed_string = process_string(string)
        f.write(processed_string + "\n")
