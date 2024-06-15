with open('../../../dataset/qq/qq_train_src.txt', 'r', encoding='utf-8') as file:
    train_lines_src = file.readlines()
with open('../../../dataset/qq/qq_train_trg.txt', 'r', encoding='utf-8') as file:
    train_lines_trg = file.readlines()
    
train_s=[]
for i in range(len(train_lines_src)):
    new_dict={}
    new_dict['source']=train_lines_src[i]
    new_dict['target']=train_lines_trg[i]
    train_s.append(new_dict)


with open('../../../dataset/qq/qq_val_src.txt', 'r', encoding='utf-8') as file:
    val_lines_src = file.readlines()
with open('../../../dataset/qq/qq_val_trg.txt', 'r', encoding='utf-8') as file:
    val_lines_trg = file.readlines()
    

val_s=[]
for i in range(len(val_lines_src)):
    new_dict={}
    new_dict['source']=val_lines_src[i]
    new_dict['target']=val_lines_trg[i]
    val_s.append(new_dict)


qq_train_data={}
qq_train_data['train']=train_s
qq_train_data['test']=val_s


import json
with open('qq_train_t5.json', 'w', encoding='utf-8') as f:
    json.dump(qq_train_data, f, ensure_ascii=False, indent=4)