import pandas as pd
import numpy as np
import torch
from scipy.linalg import logm


def fro(a,b):
    dis = np.linalg.norm(a-b, 'fro')
    return dis

def spd_dis(A,B):
    matrix_ABA = torch.matmul(torch.matmul(tensor_power(A,-0.5),B),tensor_power(A,-0.5))
    S,U = torch.linalg.eigh(matrix_ABA)
    S_trans = (torch.log(S))**2
    dis = S_trans.sum(1)

    return dis

def tensor_power(A,r):
    S, U = torch.linalg.eigh(A)
    pow_S = S**r
    power_A = torch.matmul(torch.matmul(U,torch.diag_embed(pow_S)),U.transpose(1,2))  

    return(power_A)

spds_true = pd.read_csv('data/condition/data_true.csv')
spds_class_ddpm = pd.read_csv('data/condition/data_3.csv')
spds_ddpm = pd.read_csv('data/condition/data_2.csv')
spds_frechet = pd.read_csv('data/condition/data_1.csv')


n = len(spds_true)
m = 10
n_test = 1100
a = 0

spds_true = spds_true.iloc[0:n_test,:]
spds_ddpm = spds_ddpm.iloc[0:n_test,:]
spds_frechet = spds_frechet.iloc[0:n_test,:]
spds_class_ddpm = spds_class_ddpm.iloc[0:n_test,:]

missing_rows = spds_ddpm[spds_ddpm.isnull().any(axis=1)].index

spds_ddpm= spds_ddpm.drop(missing_rows, errors='ignore')
spds_frechet = spds_frechet.drop(missing_rows, errors='ignore')
spds_true = spds_true.drop(missing_rows, errors='ignore')
spds_class_ddpm = spds_class_ddpm.drop(missing_rows, errors='ignore')

spds_true = spds_true.to_numpy()
spds_ddpm = spds_ddpm.to_numpy()
spds_frechet = spds_frechet.to_numpy()
spds_class_ddpm = spds_class_ddpm.to_numpy()

n_test2 = len(spds_ddpm)
true = spds_true.reshape(n_test2, m, m)
ddpm = spds_ddpm.reshape(n_test2, m, m)
frechet = spds_frechet.reshape(n_test2, m, m)
class_ddpm = spds_class_ddpm.reshape(n_test2, m, m)


f_dis_frechet = 0
for i in range(0,n_test2):
    f_dis_frechet = f_dis_frechet + fro(true[i,:,:],frechet[i,:,:])
f_dis_frechet = f_dis_frechet/n_test2
a_dis_frechet = spd_dis(torch.tensor(true),torch.tensor(frechet)).mean()


f_dis_ddpm = 0
for i in range(0,n_test2):
    f_dis_ddpm = f_dis_ddpm + fro(true[i,:,:],ddpm[i,:,:])
f_dis_ddpm = f_dis_ddpm/n_test2
a_dis_ddpm = spd_dis(torch.tensor(true),torch.tensor(ddpm)).mean()

f_dis_class = 0
f_dis_class= f_dis_class/n_test2
a_dis_class_list = spd_dis(torch.tensor(true),torch.tensor(class_ddpm))
cleaned_data = a_dis_class_list [~torch.isnan(a_dis_class_list )]
cleaned_data = cleaned_data[~torch.isinf(cleaned_data)]
a_dis_class = cleaned_data.mean()



print(a_dis_frechet)
print(f_dis_frechet)

print(a_dis_ddpm)
print(f_dis_ddpm)

print(a_dis_class)
print(f_dis_class)




