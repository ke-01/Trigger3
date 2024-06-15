import json
def read_list_from_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)
    
    
train_part_src='qq_train_part.txt'
with open(train_part_src,'r',encoding='utf8')as fr:
    train_src = fr.readlines()

train_part_src='qq_train_part_trg.txt'
with open(train_part_src,'r',encoding='utf8')as fr:
    train_trg = fr.readlines()
    

common_list = list(set(train_src) & set(train_trg))

indices_to_remove = [index for index, value in enumerate(train_src) if value in common_list]

train_src = [value for index, value in enumerate(train_src) if index not in indices_to_remove]
train_trg = [value for index, value in enumerate(train_trg) if index not in indices_to_remove]

pos_items=train_src[:5000] 
neg_items=common_list[:5000]

train_src=pos_items+neg_items

ct_train_src_file='qq_train_ct_src.txt'
with open(ct_train_src_file, 'w', encoding='utf-8') as f:
    for line in train_src:
        f.write(line) 