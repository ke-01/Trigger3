import json
import Levenshtein

def read_list_from_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_list_to_file(lst, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lst, f, ensure_ascii=False)
        
        
def threshold_binarize(data, threshold):
    binary_list = [1 if value >= threshold else 0 for value in data]
    
    return binary_list


def replace_lines(file1, file2, file3,one_indices,score1,score2,score3):
    
    with open(score1, 'r', encoding='utf-8') as file:
        score1_lines = json.load(file)
    with open(score2, 'r', encoding='utf-8') as file:
        score2_lines = json.load(file)

    with open(file2, 'r', encoding='utf-8') as f:
        lines_file2 = f.readlines()
    
    with open(file1, 'r', encoding='utf-8') as f:
        lines_file1 = f.readlines()
    for idx in one_indices:
        lines_file1[idx] = lines_file2[idx]
    for idx in one_indices:
        score1_lines[idx] = score2_lines[idx]

    with open(file3, 'w', encoding='utf-8') as f:
        f.writelines(lines_file1)
    save_list_to_file(score1_lines,score3)


# e.g. lt gector
score_path="qq_score_qq_lt_gector_epoch_10.txt"

with open(score_path, 'r', encoding='utf-8') as file:
    score_list = json.load(file)
processed_list = [ 1-x for x in score_list]


threshold = 0.3
binary_result = threshold_binarize(processed_list, threshold)
change_one_indices = [i for i, value in enumerate(binary_result) if value == 1]


#gector
file1 = "./qq_test_gector.txt"
file2 = "./qq_qwen_ft_gector.txt"
file3= "./fusion/qq_gector_qwen.txt"
score1="./scores/qq_score_qq_ft_gector_epoch_10.txt"
score2="./scores/qq_score_qq_ft_qwen_gector_epoch_10.txt"
score3="./scores/qq_ft_gector_epoch_10.txt"


replace_lines(file1, file2, file3 ,change_one_indices,score1,score2,score3)

