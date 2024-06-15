# when small model is GECToR and LLM is Qwen

# CT
python ct_test.py --model_type gector --gpus 0 --epochs 10

# check filter 
python filter_query.py 

# small model test 
Go to /GECToR, .. /BART, .. /mT5 folder to test

# LT
python lt_test.py --model_type gector --gpus 0 --epochs 10

# check filter 
python filter_query.py 

# LLM test 
python ../Qwen/qwen_ft.py --test_file ../Qwen/dataset/qq_test_src.txt --small_pre_file ../Qwen/dataset/qq_test_gector.txt  --output_file ./qq_qwen_ft_gector.txt

# FT
python ft_test.py --model_type gector --gpus 0 --epochs 10


# check filter 
python filter_query.py 
