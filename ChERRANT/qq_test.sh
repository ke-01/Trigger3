INPUT_FILE=./qq_train_part/qq_test_src.txt

# s2e
# OUTPUT_FILE=./qq_test_res/qq_test_gector.txt

# s2s
# OUTPUT_FILE=./qq_test_res/qq_test_bart.txt

# t5
# OUTPUT_FILE=./qq_test_res/qq_test_t5.txt

# qwen
# OUTPUT_FILE=./qq_test_res/qq_qwen_ft_t5.txt

# baichuan
# OUTPUT_FILE=./qq_test_res/qq_baichuan_ft_t5.txt

# fusion
OUTPUT_FILE=./qq_test_res/qq_test_gector_qwen_final.txt
# OUTPUT_FILE=./qq_test_res/qq_test_bart_qwen_final.txt
# OUTPUT_FILE=./qq_test_res/qq_test_t5_qwen_final.txt


HYP_PARA_FILE=./qq_test_res/qq_train.test.hyp.para
HYP_M2_FILE=./qq_test_res/qq_train.hyp.m2.char

# ref can be constructed base on qq_test_trg.txt 
REF_M2_FILE=./qq_test_res/qq_train_part.ref.m2.char

paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $HYP_PARA_FILE  

python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char  

# for different model change the -file_name
python compare_m2_for_trigger.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE -file_name qq_train_trigger_gector_



