from scipy.linalg import fractional_matrix_power
import torch


def spd_dis(A,B):
    matrix_ABA = torch.matmul(torch.matmul(tensor_power(A,-0.5),B),tensor_power(A,-0.5))
    S,U = torch.linalg.eigh(matrix_ABA)
    S_trans = (torch.log(S))**2

    dis = S_trans.sum(1)

    return dis

def spd_plus(A,B):
    matrix_plus = expm(logm(A) + logm(B))

    return matrix_plus

def spd_minus(A,B):
    matrix_minus = expm(logm(A) - logm(B))

    return matrix_minus

def spd_mul(A,r):
    matrix_mul = expm(r * logm(A))

    return matrix_mul

def Log(Y,Z):

    log_YZ = logm(torch.matmul(torch.matmul(tensor_power(Y,-0.5),Z),tensor_power(Y,-0.5)))
    result = torch.matmul(torch.matmul(tensor_power(Y,0.5),log_YZ),tensor_power(Y,0.5))

    return result

def logm(Y):
    S, U = torch.linalg.eigh(Y)
    log_S = torch.log(S)
    log_Y = torch.matmul(torch.matmul(U,torch.diag_embed(log_S)),U.transpose(1,2))
    return log_Y

def Exp(Y,Z):
    exp_YZ = expm(torch.matmul(torch.matmul(tensor_power(Y,-0.5),Z),tensor_power(Y,-0.5)))
    result = torch.matmul(torch.matmul(tensor_power(Y,0.5),exp_YZ),tensor_power(Y,0.5))

    return result

def expm(Y):
    S, U = torch.linalg.eigh(Y)
    exp_S = torch.exp(S)
    exp_Y = torch.matmul(torch.matmul(U,torch.diag_embed(exp_S)),U.transpose(1,2))
    return exp_Y

def tensor_power(A,r):
    S, U = torch.linalg.eigh(A)
    pow_S = S**r
    power_A = torch.matmul(torch.matmul(U,torch.diag_embed(pow_S)),U.transpose(1,2))

    return(power_A)

