# model-path:
# original: liuhaotian/llava-v1.5-13b
# 13b 25% flops: zwt123home123/llava-1.5-13b-prune-zp25
# 13b 12% flops: zwt123home123/llava-1.5-13b-prune-zp12
# 7b 25% flops: zwt123home123/llava-1.5-7b-prune-zp25
# 7b 12% flops: zwt123home123/llava-1.5-7b-prune-zp12
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.cli \
    --model-path zwt123home123/llava-1.5-13b-prune-zp25  \
    --image-file "test.png" \
#    --load-4bit
