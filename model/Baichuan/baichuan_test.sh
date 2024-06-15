# for gector
python baichuan_ft.py --test_file ./dataset/qq_test_src.txt --small_pre_file ./dataset/qq_test_gector.txt  --output_file ./qq_baichuan_ft_gector.txt


# for bart
python baichuan_ft.py --test_file ./dataset/qq_test_src.txt --small_pre_file ./dataset/qq_test_bart.txt  --output_file ./qq_baichuan_ft_bart.txt


# for mt5
python baichuan_ft.py --test_file ./dataset/qq_test_src.txt --small_pre_file ./dataset/qq_test_t5.txt  --output_file ./qq_baichuan_ft_t5.txt
