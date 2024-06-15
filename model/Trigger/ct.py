import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW,get_linear_schedule_with_warmup
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--data_type', type=str, default="tp", help='bert type')
parser.add_argument('--model_type', type=str, default="gector", help='model type')
parser.add_argument('--gpus', type=str, default="0", help='dataset choice')
parser.add_argument('--epochs', type=int, default=10, help='dataset choice')
parser.add_argument('--model_save_path', type=str,  default='', help='query_path')
args = parser.parse_args()

save_path="qq_ct_"+args.model_type+"_epoch_"+str(args.epochs)+".bin"



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./bert-base-chinese', num_labels=2) 
train_src_file='ct_qq.txt'

with open(train_src_file, 'r', encoding='utf-8') as file:
    train_src = file.readlines()
    

train_labels = [1] * int(len(train_src)/2) + [0] *int(len(train_src)/2)

assert len(train_labels) == len(train_src) 


src_np = np.array(train_src)
labels_np = np.array(train_labels)

random_indices = np.random.permutation(len(train_labels))

src_np = src_np[random_indices]
labels_np =labels_np[random_indices]

train_src = src_np.tolist()
train_labels= labels_np.tolist()


inputs = tokenizer(train_src, padding=True, truncation=True, return_tensors="pt")

dataset = TensorDataset(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], torch.tensor(train_labels))
train_loader = DataLoader(dataset, batch_size=256)

optimizer = AdamW(model.parameters(),
                lr = 1e-5, 
                eps = 1e-8 
                )


loss_fn = torch.nn.CrossEntropyLoss()

device=torch.device('cuda:'+args.gpus) 
model.to(device)


epochs = args.epochs
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

loss_values = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    i=0
    for batch in tqdm(train_loader):
        input_ids, token_type_ids, attention_mask, labels = [t.to(device) for t in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,labels=labels)
        
        loss=outputs[0]
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
    loss_values.append(total_loss)
    print("Epoch {} - Average Loss: {}".format(epoch+1, total_loss / len(train_loader)))


model.save_pretrained(save_path)
