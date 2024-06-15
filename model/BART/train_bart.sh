# process data
DATA_DIR=./exp_data/qq

TRAIN_SRC_FILE=../../dataset/qq/qq_train_src.txt  
TRAIN_TGT_FILE=../../dataset/qq/qq_train_trg.txt
if [ ! -f $DATA_DIR"/train.json" ]; then
    python ./utils.py $TRAIN_SRC_FILE $TRAIN_TGT_FILE $DATA_DIR"/train.json"
fi

VALID_SRC_FILE=../../data/qq/qq_val_src.txt  
VALID_TGT_FILE=../../data/qq/qq_val_trg.txt

if [ ! -f $DATA_DIR"/valid.json" ]; then
    python ./utils.py $VALID_SRC_FILE $VALID_TGT_FILE $DATA_DIR"/valid.json"
fi

# train
SEED=1
PRETRAIN_MODEL=fnlp/bart-large-chinese
MODEL_DIR=./exps/qq_bart-large-chinese
TASK_NAME=gec
CUDA_DEVICE=0,1,2,3

mkdir -p $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak
cp ./pipeline.sh $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak
cp train.py $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak
cp predict.py $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
    --do_train \
    --do_eval \
    --model_path $PRETRAIN_MODEL \
    --save_path $MODEL_DIR \
    --task $TASK_NAME \
    --data_dir $DATA_DIR \
    --seed $SEED \


# test
MODEL_PATH=./exps/qq_bart-large-chinese/gec/run_1

RESULT_DIR=$MODEL_PATH/results
mkdir -p $RESULT_DIR
INPUT_FILE=../../data/qq/qq_test_src.txt 
OUTPUT_FILE=$RESULT_DIR"/qq_test_bart.txt" 

CUDA_VISIBLE_DEVICES=0 python -u predict.py \
    --model_path $MODEL_PATH \
    --input_path $INPUT_FILE \
    --output_path $OUTPUT_FILE ;


# for trigger's train dataset
MODEL_PATH=./exps/qq_bart-large-chinese/gec/run_1

RESULT_DIR=$MODEL_PATH/results
mkdir -p $RESULT_DIR
INPUT_FILE=../../data/qq/qq_train_part.txt 
OUTPUT_FILE=$RESULT_DIR"/qq_trigger_train_bart.txt" 

CUDA_VISIBLE_DEVICES=0 python -u predict.py \
    --model_path $MODEL_PATH \
    --input_path $INPUT_FILE \
    --output_path $OUTPUT_FILE ;

# for trigger's val dataset
MODEL_PATH=./exps/qq_bart-large-chinese/gec/run_1

RESULT_DIR=$MODEL_PATH/results
mkdir -p $RESULT_DIR
INPUT_FILE=../../data/qq/qq_val_src.txt 
OUTPUT_FILE=$RESULT_DIR"/qq_trigger_val_bart.txt" 

CUDA_VISIBLE_DEVICES=0 python -u predict.py \
    --model_path $MODEL_PATH \
    --input_path $INPUT_FILE \
    --output_path $OUTPUT_FILE ;
