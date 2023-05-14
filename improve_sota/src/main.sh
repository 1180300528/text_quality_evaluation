## 第一步，对NEZHA模型进行继续预训练（MLM任务）。
#python pretrain.py \
#            --EPOCH 10 \
#            --BATCH_SIZE 8 \
#            --seed 1997 \
#            --model_name "nezha-base-cn"
#
## 第二步，进行FineTune。加入统计信息，训练NEZHA模型，数据为10折中的第6,7,8,9折。其他参数如下所示。
#python train.py \
#            --BATCH_SIZE 8 \
#            --EPOCH 3 \
#            --DEVICE 'cuda:0' \
#            --model_name "nezha-base-cn" \
#            --k_fold 10 \
#            --seed 1997 \
#            --lr 2e-5 \
#            --fold_list 6 7 8 9 \
#            --input_format "statistics+320+64" \
#            --output_prefix "model_statistics_nezha"
#
#
## 第三步，进行FineTune。加入统计信息，训练NEZHA模型，不分k折，进行所有数据的finetune。其他参数如下所示。
#python train.py \
#            --BATCH_SIZE 8 \
#            --EPOCH 3 \
#            --DEVICE 'cuda:0' \
#            --model_name "nezha-base-cn" \
#            --seed 1997 \
#            --lr 2e-5 \
#            --all_data True \
#            --input_format "statistics+320+64" \
#            --output_prefix "model_statistics_nezha"
#
#
## 第四步，进行FineTune。加入统计信息，训练XLNET模型，数据为10折模型的第0、1折。其他参数如下所示。（由于机器显存不足所以batchsize为4。若觉得太慢，可增大batch_size，但记得同步调整learning rate）
#python train.py \
#            --BATCH_SIZE 4 \
#            --EPOCH 3 \
#            --DEVICE 'cuda:0' \
#            --model_name "xlnet-base-cn" \
#            --k_fold 10 \
#            --seed 1997 \
#            --lr 1e-5 \
#            --max_seq_len 768 \
#            --fold_list 0 1 \
#            --input_format "statistics+512+128" \
#            --output_prefix "model_statistics_xlnet"
#
#
## 第五步，进行FineTune。加入统计信息，训练NEZHA模型，数据为10折模型的第0、1、2、3折。其他参数如下所示。
#python train.py \
#            --BATCH_SIZE 8 \
#            --EPOCH 3 \
#            --DEVICE 'cuda:0' \
#            --model_name "nezha-base-cn" \
#            --k_fold 10 \
#            --seed 1997 \
#            --lr 2e-5 \
#            --gamma 2 \
#            --is_fgm True \
#            --fold_list 0 1 2 3 \
#            --input_format "200+entities" \
#            --output_prefix "model_nezha_entities"


# 第六步，根据训练好的11个模型进行预测，预测出11份结果。本部分是预测部分的第一部分。
python predict.py \
            --BATCH_SIZE 8 \
            --DEVICE 'cuda:0' \
            --model_name "nezha-base-cn" \
            --model_list "trained_model/model_statistics_nezha_6.pth" "trained_model/model_statistics_nezha_7.pth" \
                        "trained_model/model_statistics_nezha_8.pth" "trained_model/model_statistics_nezha_9.pth" "trained_model/model_statistics_nezha_all.pth" \
            --input_format "statistics+320+64"


# 第七步，根据训练好的11个模型进行预测，预测出11份结果。本部分是预测部分的第二部分。
python predict.py \
            --BATCH_SIZE 8 \
            --DEVICE 'cuda:0' \
            --model_name "xlnet-base-cn" \
            --model_list "trained_model/model_statistics_xlnet_0.pth" "trained_model/model_statistics_xlnet_1.pth" \
            --input_format "statistics+768+128"


# 第八步，根据训练好的11个模型进行预测，预测出11份结果。本部分是预测部分的第三部分。
python predict.py \
            --BATCH_SIZE 8 \
            --DEVICE 'cuda:0' \
            --model_name "nezha-base-cn" \
            --model_list "trained_model/model_nezha_entities_0.pth" "trained_model/model_nezha_entities_1.pth" \
                    "trained_model/model_nezha_entities_2.pth" "trained_model/model_nezha_entities_3.pth" \
            --input_format "200+entities"


# 第九步，根据上述11份结果进行Voting融合。
python voting.py \
            --dir_name "output" \
            --is_filter True

