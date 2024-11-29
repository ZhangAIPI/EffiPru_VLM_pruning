# :rocket: YOPO: You Only Prune Once for Your MLLMs


This repository contains the core code to implement the idea of  our paper:

[Treat Visual Tokens as Text? But Your MLLM Only Needs Fewer Efforts to See](https://arxiv.org/abs/2410.06169v2)

![Alt text](images/method_fig.png "Overview of our method.")

### :bangbang: While many studies focus on pruning visual tokens to reduce the computational overhead caused by visual redundancy, the process of identifying these tokens for each conversation is itself resource-intensive. Now the question comes, ###  

### **Can we prune our MLLM just once instead?**:interrobang: ###



## Install
1. Set up LLavA  https://github.com/haotian-liu/LLaVA 
```Shell
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

2. Copy our updated `modeling_llama.py` to transformer library
```Shell
cp ../modeling_llama_prune.py {YOUR ENV PATH}/lib/python3.10/site-packages/transformers//models/llama/modeling_llama.py
# eg. cp ../modeling_llama_prune.py /opt/conda/envs/llava/lib/python3.10/site-packages/transformers//models/llama/modeling_llama.py

```
## Inference
1. Download the checkpoints of pruned LLaVA
   
   [LLaVA-1.5-7B (12% FLOPs)](https://huggingface.co/zwt123home123/llava-1.5-7b-prune-zp12)

   [LLaVA-1.5-7B (25% FLOPs)](https://huggingface.co/zwt123home123/llava-1.5-13b-prune-zp25)

   [LLaVA-1.5-13B (12% FLOPs)](https://huggingface.co/zwt123home123/llava-1.5-7b-prune-zp12)

   [LLaVA-1.5-13B (25% FLOPs)](https://huggingface.co/zwt123home123/llava-1.5-13b-prune-zp25)

2. Run inference
```Shell
bash infer.sh
```
:triangular_flag_on_post:  We will release the code for pruning the visual computation in Qwen, InternVL without the fine-tuning process very soon.  
 
## Training

1. Download and set up LLaVA-1.5 2nd stage training data
   https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md
2. Download LLaVA-1.5 mm_projector weights
   
   https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5

   https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5

   put them into `./checkpoints/llava-v1.5-13b-pretrain` and `./checkpoints/llava-v1.5-7b-pretrain` respectively
4. Run training
```Shell
bash scripts/v1_5/finetune_yopo.sh
```
## Evaluation

## Citation

If you find the idea or code useful for your research, please consider citing our [paper](https://arxiv.org/abs/2403.12777):

```
@article{zhang2024treat,
  title={Treat Visual Tokens as Text? But Your MLLM Only Needs Fewer Efforts to See},
  author={Zhang, Zeliang and Pham, Phu and Zhao, Wentian and Wan, Kun and Li, Yu-Jhe and Zhou, Jianing and Miranda, Daniel and Kale, Ajinkya and Xu, Chenliang},
  journal={arXiv preprint arXiv:2410.06169},
  year={2024}
}
```

## Contact
Questions and suggestions can be sent to hust0426@gmail.com and {wezhao, kuwan}@adobe.com.
