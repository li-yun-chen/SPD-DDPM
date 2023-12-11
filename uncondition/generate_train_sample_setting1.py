from support_function import *
import torch
import numpy as np
import pandas as pd
from pyriemann.datasets import sample_gaussian_spd



def generate_train_sample(n,m):
    init = torch.tensor(generate_init(m)).unsqueeze(0).to("cuda")
    sample_list = torch.empty((n, m, m))
    mean = np.eye(m)
    samples = sample_gaussian_spd(n,mean,0.4,n_jobs=40)
    samples = torch.tensor(samples).to("cuda")
    sample_beta = tensor_power(samples,0.5)
    sample_list = torch.matmul(torch.matmul((tensor_power(init,0.5)),samples),tensor_power(init,0.5))
    vectors = sample_list.reshape(n, m*m)

    return vectors.cpu(),init.cpu()

def generate_init(m):
    A = np.random.rand(m, m)
    A = 0.5 * (A + A.T)
    eigenvalues, _ = np.linalg.eig(A)
    A += np.eye(m) * (np.abs(np.min(eigenvalues)) + 1e-8)
    return A

n = 15000
m = 8
a,b = generate_train_sample(n,m)
df = pd.DataFrame(a)
df.to_csv("data/uncondition/exp1_setting1_train_sample.csv",index=False)
init_sam = pd.DataFrame(b[0])
init_sam.to_csv("data/uncondition/exp1_setting1_init.csv",index=False)
