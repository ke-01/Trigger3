# process data
python dataset/data_process.py

# train
python pretrain.py --correction_data_path ./dataset/qq_train_t5.json --save_file_name qq_train_t5 

# test
python prediction.py --save_file_name qq_train_t5 --test_file ./dataset/qq_test_src.txt --output_file ./qq_test_t5.txt

# for trigger's train dataset
python prediction.py --save_file_name qq_train_t5 --test_file ./dataset/qq_train_part.txt --output_file ./qq_trigger_train_t5.txt

# for trigger's val dataset
python prediction.py --save_file_name qq_train_t5 --test_file ./dataset/qq_val_src.txt --output_file ./qq_trigger_val_t5.txt