import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW
import json
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--data_type', type=str, default="qq", help='bert type')
parser.add_argument('--model_type', type=str, default="gector", help='model type')
parser.add_argument('--gpus', type=str, default="0", help='dataset choice')
parser.add_argument('--epochs', type=int, default=10, help='dataset choice')

parser.add_argument('--model_save_path', type=str,  default='', help='query_path')
parser.add_argument('--output_save_path', type=str,  default='', help='query_path')
args = parser.parse_args()

save_path="qq_ct_"+args.model_type+"_epoch_"+str(args.epochs)+".bin"

score_output_path=args.data_type+"score_"+"qq_ct_"+"_"+args.model_type+"_epoch_"+str(args.epochs)+".txt"

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(save_path, num_labels=2) 


src_file_path = "./qq_test_src.txt"
with open(src_file_path, 'r', encoding='utf-8') as file:
    src_lines = file.readlines()


inputs = tokenizer(src_lines,  padding=True, truncation=True, return_tensors="pt")

dataset = TensorDataset(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
test_loader = DataLoader(dataset, batch_size=1)


device=torch.device('cuda:'+args.gpus) 
model.to(device)

model.eval()  

def save_list_to_file(lst, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False)

res=[]
scores=[]
with torch.no_grad():
    for batch in tqdm(test_loader):
        input_ids, token_type_ids, attention_mask = [t.to(device) for t in batch]
        
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        score=F.softmax(logits,dim=1)[0][0]

        scores.append(score.item())

save_list_to_file(scores, score_output_path)




