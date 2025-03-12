# create meta info
VIDEO_DIR=$1
current_time=$(date "+%Y%m%d_%H%M%S")
META_INFO_PATH="./eval_results/${current_time}/results.json"
META_INFO_DIR=$(dirname "$META_INFO_PATH")
python bench_utils/create_meta_info.py -v $VIDEO_DIR -o $META_INFO_PATH

# PAS metric
python perceptible_amplitude_score.py --meta_info_path $META_INFO_PATH \
    --box_threshold 0.25 \
    --text_threshold 0.20 \
    --grid_size 30 \
    --device cuda

# OIS metric
python object_integrity_score.py --meta-info-path $META_INFO_PATH \
    --save-predictions

# TCS metric
python temporal_coherence_score.py --meta_info_path $META_INFO_PATH \
    --box_threshold 0.25 \
    --text_threshold 0.20 \
    --grid_size 50

# CAS metric
set -x  # print the commands
OUTPUT_DIR=$META_INFO_DIR  # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH=$META_INFO_DIR  # The data list folder.
MODEL_PATH='./.cache/vit_g_vmbench.pt'  # Model for initializing parameters
GPUS_PER_NODE=1  # used GPU numbers
MASTER_PORT=16888
OMP_NUM_THREADS=1 torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port ${MASTER_PORT}\
        commonsense_adherence_score.py \
        --model vit_giant_patch14_224 \
        --data_set Commonsense-Adherence \
        --nb_classes 5 \
        --meta_info_path ${META_INFO_PATH} \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 10 \
        --input_size 224 \
        --num_workers 10 \
        --drop_path 0.3 \
        --dist_eval --enable_deepspeed --eval

# MSS metric
python motion_smoothness_score.py --meta_info_path $META_INFO_PATH # Note: this metric score is based on the PAS, so calculate PAS first

# save evaluation results
python bench_utils/calculate_score.py -i $META_INFO_PATH -o $META_INFO_DIR"/scores.csv"
# save evaluation results
python bench_utils/calculate_score.py -i $META_INFO_PATH -o $META_INFO_DIR"/scores.csv"