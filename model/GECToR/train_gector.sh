# process data

## Download Structbert
if [ ! -f ./plm/chinese-struct-bert-large/pytorch_model.bin ]; then
    wget https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/StructBERT/ch_model
    mv ch_model ./plm/chinese-struct-bert-large/pytorch_model.bin
fi

## Tokenize
SRC_FILE=../../dataset/qq/qq_train_src.txt  
TGT_FILE=../../dataset/qq/qq_train_trg.txt  
if [ ! -f $SRC_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $SRC_FILE > $SRC_FILE".char"  
fi
if [ ! -f $TGT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $TGT_FILE > $TGT_FILE".char"  
fi

## Generate label file
LABEL_FILE=../../dataset/qq/qq_train.label  
if [ ! -f $LABEL_FILE ]; then
    python ./utils/preprocess_data.py -s $SRC_FILE".char" -t $TGT_FILE".char" -o $LABEL_FILE --worker_num 32
    shuf $LABEL_FILE > $LABEL_FILE".shuf"
fi

SRC_FILE=../../dataset/qq/qq_val_src.txt  
TGT_FILE=../../dataset/qq/qq_val_trg.txt  
if [ ! -f $SRC_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $SRC_FILE > $SRC_FILE".char"  
fi
if [ ! -f $TGT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $TGT_FILE > $TGT_FILE".char"  
fi

## Generate label file
DEV_SET=../../dataset/qq/qq_val.label  
if [ ! -f $DEV_SET ]; then
    python ./utils/preprocess_data.py -s $SRC_FILE".char" -t $TGT_FILE".char" -o $DEV_SET --worker_num 32
    shuf $DEV_SET > $DEV_SET".shuf"
fi


# train
CUDA_DEVICE=0
SEED=1

LABEL_FILE=../../data/qq/qq_train.label.shuf  
DEV_SET=../../data/qq/qq_val.label.shuf
MODEL_DIR=./exps/qq
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi

PRETRAIN_WEIGHTS_DIR=./plm/chinese-struct-bert-large

mkdir ${MODEL_DIR}/src_bak
cp ./pipeline.sh $MODEL_DIR/src_bak
cp -r ./gector $MODEL_DIR/src_bak
cp ./train.py $MODEL_DIR/src_bak
cp ./predict.py $MODEL_DIR/src_bak

VOCAB_PATH=./data/output_vocabulary_chinese_char_qq

## Freeze encoder (Cold Step)
COLD_LR=1e-3
COLD_BATCH_SIZE=128
COLD_MODEL_NAME=Best_Model_Stage_1
COLD_EPOCH=2

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 0\
                --train_set $LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $COLD_MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $COLD_BATCH_SIZE\
                --n_epoch $COLD_EPOCH\
                --lr $COLD_LR\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --seed $SEED

## Unfreeze encoder
LR=1e-5
BATCH_SIZE=64
ACCUMULATION_SIZE=4
MODEL_NAME=Best_Model_Stage_2
EPOCH=20
PATIENCE=3

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py --tune_bert 1\
                --train_set $LABEL_FILE".shuf"\
                --dev_set $DEV_SET\
                --model_dir $MODEL_DIR\
                --model_name $MODEL_NAME\
                --vocab_path $VOCAB_PATH\
                --batch_size $BATCH_SIZE\
                --n_epoch $EPOCH\
                --lr $LR\
                --accumulation_size $ACCUMULATION_SIZE\
                --patience $PATIENCE\
                --weights_name $PRETRAIN_WEIGHTS_DIR\
                --pretrain_folder $MODEL_DIR\
                --pretrain "Temp_Model"\
                --seed $SEED


# test
MODEL_PATH=$MODEL_DIR"/Best_Model_Stage_2.th"
RESULT_DIR=$MODEL_DIR"/results"

INPUT_FILE=../../dataset/qq/qq_test_src.txt 

if [ ! -f $INPUT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $INPUT_FILE > $INPUT_FILE".char"  # 分字
fi
if [ ! -d $RESULT_DIR ]; then
  mkdir -p $RESULT_DIR
fi
OUTPUT_FILE=$RESULT_DIR"/qq_test_gector.txt"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --input_file $INPUT_FILE".char"\
                  --output_file $OUTPUT_FILE --log


# for trigger's train dataset
INPUT_FILE=../../dataset/qq/qq_train_part.txt 

if [ ! -f $INPUT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $INPUT_FILE > $INPUT_FILE".char"  # 分字
fi
OUTPUT_FILE=$RESULT_DIR"/qq_trigger_train_gector.txt"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --input_file $INPUT_FILE".char"\
                  --output_file $OUTPUT_FILE --log

# for trigger's val dataset
INPUT_FILE=../../dataset/qq/qq_val_src.txt 

if [ ! -f $INPUT_FILE".char" ]; then
    python ../../tools/segment/segment_bert.py < $INPUT_FILE > $INPUT_FILE".char"  # 分字
fi
OUTPUT_FILE=$RESULT_DIR"/qq_trigger_val_gector.txt"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python predict.py --model_path $MODEL_PATH\
                  --weights_name $PRETRAIN_WEIGHTS_DIR\
                  --vocab_path $VOCAB_PATH\
                  --input_file $INPUT_FILE".char"\
                  --output_file $OUTPUT_FILE --log