import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
import logging
from support_function import *
from SPD_net import SPD_NET
import warnings
import numpy as np
import json
from datetime import datetime
from pyriemann.datasets import sample_gaussian_spd
import math
warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")



def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


def remove_subtensors_with_nan(input_tensor):
    assert len(input_tensor.shape) == 3, "The input tensor should have three dimensions (n*m*m)"
    mask = torch.isnan(input_tensor).view(input_tensor.shape[0], -1).any(dim=-1)
    output_tensor = input_tensor[~mask]
    return output_tensor


class Diffusion:
    def __init__(self, noise_steps=200, beta_start=1e-4, beta_end=0.08 ,spd_size=20, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.device = device
        self.spd_size = spd_size
        self.beta = self.prepare_noise_schedule().to(device)   
        self.alpha = torch.sqrt(1 - self.beta)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)    
        self.beta_hat = 1-self.alpha_hat**2           

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        alpha_hat = self.alpha_hat[t]
        alpha = self.alpha[t]
        beta_hat = torch.sqrt(self.beta_hat[t])
        beta = self.beta[t]
        alpha_t_1 = self.alpha[t-1]
        beta_hat_t_1 = self.beta_hat[t-1]
        n = x.shape[0]
        m = x.shape[1]

        mean = np.eye(m)
        epi1 = sample_gaussian_spd(n,mean,1,n_jobs=40)
        epi1 = torch.tensor(epi1).to("cuda")

        epi1_beta = spd_mul(epi1,beta_hat.unsqueeze(1).unsqueeze(2))

        x_t = spd_plus(spd_mul(x,alpha_hat.unsqueeze(1).unsqueeze(2)),spd_mul(epi1,beta_hat.unsqueeze(1).unsqueeze(2)))

        S,U= torch.linalg.eigh(x_t)
        S_log = torch.log(S)
        r1 = (beta)/(beta_hat)/alpha
 
        return x_t ,epi1 ,r1

    def sample_timesteps(self, n):
        t =torch.randint(low=1, high=self.noise_steps, size=(n,))
        return t

    def sample(self, model,n):
        init = pd.read_csv("data/uncondition/exp1_setting1_init.csv")
        init =  torch.tensor(init.values)
        init_tensor =  init.repeat(n, 1,1).to("cuda")
        model.eval()
        with torch.no_grad():
            mean = np.eye(self.spd_size)
            x = sample_gaussian_spd(n,mean,1,n_jobs=40)
            x = torch.tensor(x).to("cuda")
            test_dis = spd_dis(init_tensor,x).mean().cpu()
            print(test_dis)
            for i in reversed(range(1,self.noise_steps)):
                print(i)
                t = torch.tensor([i]).repeat(n).to("cuda") 
                beta = self.beta[t][0]
                beta_hat = torch.sqrt(self.beta_hat[t][0])
                beta_hat_t_1 = self.beta_hat[t-1][0]
                alpha = self.alpha[t][0]
                alpha_t_1 = self.alpha[t-1][0]
                alpha_hat = self.alpha_hat[t][0]
                alpha_hat_t_1 = self.alpha_hat[t-1][0]

                n  = x.shape[0]
                mean = np.eye(self.spd_size)
                epi = sample_gaussian_spd(n,mean,1,n_jobs=40)
                epi = torch.tensor(epi).to("cuda")
                epi_beta = tensor_power(epi,beta_hat.item())
                predicted_noise = model(x, t)
                loss = spd_dis(epi, predicted_noise).mean()

                r1 =1/alpha
                r2 =beta/beta_hat/alpha
                mu_x = spd_minus(spd_mul(x,r1),spd_mul(predicted_noise,r2))


                sigma = torch.sqrt(beta)
                epi1_sigma = tensor_power(epi,sigma)

                r = 10
                x = spd_plus(mu_x,spd_mul(epi,sigma/r))
                
                test_dis = spd_dis(init_tensor,x).mean().cpu()

                
                unit_matrix = torch.eye(self.spd_size).unsqueeze(0)  
                merged_tensor = unit_matrix.repeat(n, 1, 1).to("cuda").double()
                dis2 = spd_dis(merged_tensor,x).mean().cpu()

                print(test_dis)
                print(loss)
                print(dis2)

        model.train()   
        return x

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
    
def train(args):

    device = args.device
    dataset = CSVDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = SPD_NET(args.spd_size,args.time_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(spd_size=args.spd_size, device=device)

    epoch_start = 1
    if args.resume != '':
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))

    results = {'train_loss': [],'lr':[]}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)


    for epoch in range(epoch_start,args.epochs):
        total_loss = []
        pbar = tqdm(dataloader)
        lr1 = adjust_learning_rate(optimizer, epoch, args)
        for  i, spds in enumerate(pbar):
            spds = spds.to("cuda")
            t = diffusion.sample_timesteps(args.batch_size).to(device)
            
            x_t, epi,r = diffusion.noise_images(spds, t)
            predicted_noise = model(x_t, t)

            loss = spd_dis(predicted_noise,epi)
            loss = loss.mean()

            total_loss.append(loss.item()) 
            pbar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs, optimizer.param_groups[0]['lr'], loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(total_loss)
        results['train_loss'].append(epoch_loss)
        results['lr'].append(lr1)
        data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
        data_frame.to_csv(args.results_dir + '/log.csv', index_label='epoch')

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict(),}, args.results_dir + '/model_last.pth')


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 50
    args.batch_size = 150
    args.spd_size = 8
    args.time_size = 256
    args.dataset_path = "data/uncondition/train_data.csv"
    args.device = "cuda"
    args.lr = 0.0015
    args.results_dir = 'result/spd_ddpm_un-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.resume = ""

    #args.resume = "result/ddpm_un/model_last.pth"
    #args.results_dir = "result/ddpm_un"

    train(args)


if __name__ == '__main__':
   launch() 

