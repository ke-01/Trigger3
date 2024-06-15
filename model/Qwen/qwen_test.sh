# for gector
python qwen_ft.py --test_file ./dataset/qq_test_src.txt --small_pre_file ./dataset/qq_test_gector.txt  --output_file ./qq_qwen_ft_gector.txt


# for bart
python qwen_ft.py --test_file ./dataset/qq_test_src.txt --small_pre_file ./dataset/qq_test_bart.txt  --output_file ./qq_qwen_ft_bart.txt


# for mt5
python qwen_ft.py --test_file ./dataset/qq_test_src.txt --small_pre_file ./dataset/qq_test_t5.txt  --output_file ./qq_qwen_ft_t5.txt


# for train_trigger
python qwen_ft.py --test_file ./dataset/qq_train_part.txt --small_pre_file ./dataset/qq_train_part_gector.txt  --output_file ./qq_qwen_train_part_gector.txt