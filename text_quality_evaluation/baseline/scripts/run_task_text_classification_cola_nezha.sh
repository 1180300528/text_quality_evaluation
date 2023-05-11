CURRENT_DIR=`pwd`
export CACHE_DIR=$CURRENT_DIR/../examples/NEZHA
export MODEL_DIR=$CACHE_DIR
export DATA_DIR=$CURRENT_DIR/../dataset
export OUTPUR_DIR=$CURRENT_DIR/../outputs
export TASK_NAME=CCKS
export MODEL_TYPE=mine

#-----------training-----------------
python ../examples/task_text_classification_cola_nezha.py \
  --model_type=$MODEL_TYPE \
  --pretrained_model_path=$MODEL_DIR \
  --pretrained_cache_dir=$CACHE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --num_labels=2 \
  --device_id='1' \
  --evaluate_during_training \
  --do_lower_case \
  --experiment_code='V25_NEZHA' \
  --checkpoint_mode=max \
  --checkpoint_monitor=eval_F1 \
  --data_dir=$DATA_DIR/$TASK_NAME/ \
  --train_input_file=train.json \
  --eval_input_file=eval.json \
  --overwrite_data_cache \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --train_max_seq_length=512 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --eval_all_checkpoints \
  --learning_rate=2e-5 \
  --num_train_epochs=10 \
  --gradient_accumulation_steps=1 \
  --warmup_proportion=0.1 \
  --scheduler_type='linear' \
  --logging_steps=-1 \
  --save_steps=-1 \
  --seed=42



