# BELLE-7B-gptq: local saved model path
# Save compressed model
CUDA_VISIBLE_DEVICES=0 python bloom.py BelleGroup/BELLE-7B-2M wikitext2 --wbits 8 --groupsize 128 --save BELLE-7B-gptq/bloom7b-2m-8bit-128g.pt

