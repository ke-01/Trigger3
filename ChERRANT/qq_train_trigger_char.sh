INPUT_FILE=./qq_train_part/qq_train_part.txt
# # our

# s2e
OUTPUT_FILE=./train_trigger/qq_trigger_train_gector.txt

# s2s
# OUTPUT_FILE=./train_trigger/qq_trigger_train_bart.output2

# t5
# OUTPUT_FILE=./train_trigger/qq_trigger_train_t5.txt

# qwen
# OUTPUT_FILE=./train_trigger/qq_qwen_ft_trigger.txt

# baichuan
# OUTPUT_FILE=./train_trigger/qq_baichuan_ft_trigger.txt

HYP_PARA_FILE=./train_trigger/qq_train.test.hyp.para
HYP_M2_FILE=./train_trigger/qq_train.hyp.m2.char

# ref can be constructed base on qq_train_part_trg.txt 
REF_M2_FILE=./train_trigger/qq_train_part.ref.m2.char

paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $HYP_PARA_FILE  

python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char  

# for different model change the -file_name
python compare_m2_for_trigger.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE -file_name qq_train_trigger_gector_



