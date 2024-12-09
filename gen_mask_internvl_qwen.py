import torch
import numpy as np

model_name = "internvl2.0_8B"
# model_name = "qwen2_vl_8b"
head_num = 24 # num of heads kept on average

if model_name == "internvl2.0_26B":
    total_head_num = 48
if model_name == "internvl2.0_8B":
    total_head_num = 32
if model_name == "internvl2.0_4B":
    total_head_num = 32
if model_name == "qwen2_vl_8b":
    total_head_num = 28
    
ratio_lst=[]
for layer_idx in range(total_head_num):            
    ratio_lst.append(torch.load(model_name+"/"+str(layer_idx)+".pth").unsqueeze(0).cpu())
ratio_lst = torch.cat(ratio_lst,dim=0)

if model_name == "internvl2.0_26B":
    ratio_threshold = 0.052
if model_name == "internvl2.0_8B":
    ratio_threshold = 0.11    
if model_name == "internvl2.0_4B":
    ratio_threshold = 0.08
if model_name == "qwen2_vl_8b":
    ratio_threshold = 0.07


mask = ratio_lst>ratio_threshold # please adjust this threshold to keep different num of heads

torch.save(mask,model_name+"/"+"mask_"+str(head_num)+".pth")
