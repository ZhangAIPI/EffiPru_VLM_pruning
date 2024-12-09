import torch
import numpy as np
offset = 35
image_token_num = 576
mask_lst=[]
mask_total = np.zeros((40, 40))
for layer_idx in range(40):
    attn_weights_lst=[]
    for i in range(4):
        attn_weights_lst.append(torch.load("llava-1.5-13b/weights"+str(i)+"/"+str(layer_idx)+".pth"))
    
    attn_weights = torch.cat(attn_weights_lst,dim=0)
    
    for i in range(100):
        res_v = torch.sum(attn_weights[i,:,:,offset:offset+image_token_num],dim=[1,2]) 
        res_t = torch.sum(attn_weights[i,:,:,offset+image_token_num:],dim=[1,2])  
        res_s = torch.sum(attn_weights[i,:,:,:offset],dim=[1,2])
        res = res_v/(res_t+res_s)  
        if layer_idx>=2:
            mask = res>0.035 # adjust the threadhold, headcut10: 0.2 headcut20: 0.077 headcut30: 0.035
        else:
            mask = res>=0
        
        mask = mask.int()
        mask_total[layer_idx]+=mask.detach().cpu().numpy()
        
mask = mask_total>40 # majority voting  
np.save("llava-1.5-13b/headmask_30.npy", mask)


