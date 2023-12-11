from SPD_net import SPD_NET
import torch 
from ddpm import Diffusion
import pandas as pd
from support_function import *
import matplotlib.pyplot as plt
from pyriemann.datasets import sample_gaussian_spd


def ddpm_sample(n,path):
    device = "cuda"
    model = SPD_NET(spd_size=m,time_size=256).to(device)
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['state_dict'])
    diffusion = Diffusion(spd_size=m, device=device)
    x = diffusion.sample(model,n)
    return x

n = 300
m = 8
init = pd.read_csv("data/uncondition/exp1_setting1_init.csv")
init =  torch.tensor(init.values)
init_tensor =  init.repeat(n, 1,1).to("cuda")
model_path = "result/spd_uncondition.pth"

sample_list = ddpm_sample(n,model_path)
test_dis = spd_dis(init_tensor,sample_list).cpu()
mask1 = torch.isnan(test_dis) == False
test_dis = test_dis[mask1]

print(test_dis.mean())

vectors = sample_list.reshape(n, m*m).cpu()
df = pd.DataFrame(vectors)
df.to_csv("data/uncondition/generated_samples_spd_ddpm.csv.csv",index=False)


