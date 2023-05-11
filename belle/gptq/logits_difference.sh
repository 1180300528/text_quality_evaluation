CUDA_VISIBLE_DEVICES=0 python logits_difference.py BelleGroup/BELLE-7B-gptq --wbits 8 --groupsize 128 --load BELLE-7B-gptq/bloom7b-2m-8bit-128g.pt --text "hello" --max_length 2048
