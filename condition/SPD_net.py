import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from support_function import *

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class SPD_NET(nn.Module):
    def __init__(self, spd_size,time_size,Y_size):
        super().__init__()
        self.time_dim = time_size
        self.Y_size = Y_size
        
        self.trans1 = SPDTransform(spd_size, 10, self.time_dim,Y_size)
        self.trans1_5 = SPDTransform1(10, 10, self.time_dim,Y_size)
        self.trans2 = SPDTransform(10, 10, self.time_dim,Y_size)
        self.trans2_5 = SPDTransform1(10, 10, self.time_dim,Y_size)

        self.trans3 = SPDTransform(10, 6, self.time_dim,Y_size)
        self.trans3_5= SPDTransform1(6, 6, self.time_dim,Y_size)

        self.trans4 = SPDTransform(6, 6, self.time_dim,Y_size)
        self.trans4_5 = SPDTransform1(6, 6, self.time_dim,Y_size)

        self.trans5 = SPDTransform(6, 3, self.time_dim,Y_size)
        self.trans5_5 = SPDTransform1(3, 3, self.time_dim,Y_size) 

        self.trans5_8 = SPDTransform(3, 3, self.time_dim,Y_size)
        self.trans5_9 = SPDTransform1(3, 3, self.time_dim,Y_size) 

        self.trans6 = SPDTransform(3, 6, self.time_dim,Y_size)
        self.trans6_5 = SPDTransform1(6, 6, self.time_dim,Y_size)

        self.trans7 = SPDTransform(6, 6, self.time_dim,Y_size)
        self.trans7_5 = SPDTransform1(6, 6, self.time_dim,Y_size)

        self.trans8 = SPDTransform(6, 10, self.time_dim,Y_size)
        self.trans8_5 = SPDTransform1(10, 10, self.time_dim,Y_size)
        self.trans9 = SPDTransform(10, 10, self.time_dim,Y_size)
        self.trans9_5 = SPDTransform1(10, 10, self.time_dim,Y_size)
    
        self.trans10 = SPDTransform(10, 10, self.time_dim,Y_size)

        self.rect1  = SPDRectified()
        self.rect1_5  = SPDRectified()
        self.rect2  = SPDRectified()
        self.rect2_5  = SPDRectified()
        self.rect3  = SPDRectified()
        self.rect3_5  = SPDRectified()
        self.rect4  = SPDRectified()
        self.rect4_5  = SPDRectified()
        self.rect5  = SPDRectified()
        self.rect5_5  = SPDRectified()
        self.rect5_8  = SPDRectified()
        self.rect5_9  = SPDRectified()
        self.rect6 = SPDRectified()
        self.rect6_5 = SPDRectified()
        self.rect7 = SPDRectified()
        self.rect7_5 = SPDRectified()
        self.rect8 = SPDRectified()
        self.rect8_5 = SPDRectified()
        self.rect9 = SPDRectified()
        self.rect9_5 = SPDRectified()
        
        self.rect10 = SPDRectified()
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000** (torch.arange(0, channels, 2, device="cuda")/ channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2 )* inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2 ) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1).double()
        return pos_enc

    def unet_forwad(self, x, t,Y):
        
        r = 0.5
        x1 = self.trans1(x,t,Y)  
        x1 = self.rect1(x1)

        x1_5 = self.trans1_5(x1,t,Y)   
        x1_5= self.rect1_5(x1_5)

        x2 = self.trans2(x1_5,t,Y)  
        x2 = self.rect2(x2)
        x2_5 = self.trans2_5(x2,t,Y)  
        x2_5 = self.rect2_5(x2_5)

        x3 = self.trans3(x2_5,t,Y)  
        x3 = self.rect3(x3)
        x3_5 = self.trans3_5(x3,t,Y)  
        x3_5 = self.rect3_5(x3_5)

        x4 = self.trans4(x3_5,t,Y)  
        x4 = self.rect4(x4)
        x4_5 = self.trans4_5(x4,t,Y)  
        x4_5 = self.rect4_5(x4_5)

        x5 = self.trans5(x4_5,t,Y)  
        x5 = self.rect5(x5)
        x5_5 = self.trans5_5(x5,t,Y)  
        x5_5 = self.rect5_5(x5_5)

        x5_8 = self.trans5_8(x5_5,t,Y)  
        x5_8 = self.rect5_8(x5_8)
        x5_9 = self.trans5_9(x5_8,t,Y)  
        x5_9 = self.rect5_9(x5_9)

        x6 = (r*x5_9 + (1-r)*x5_5)/2
        x6 = self.trans6(x6,t,Y)  
        x6 = self.rect6(x6)
        x6_5 = self.trans6_5(x6,t,Y)  
        x6_5 = self.rect6_5(x6_5)

        x7 = (r*x6_5 + (1-r)*x4_5)/2  
        
        x7 = self.trans7(x7,t,Y)  
        x7 = self.rect7(x7)
        x7_5 = self.trans7_5(x7,t,Y)  
        x7_5 = self.rect7_5(x7_5)

        x8= (r*x7_5 + (1-r)*x3_5)/2
        x8 = self.trans8(x8,t,Y)  
        x8 = self.rect8(x8)
        x8_5 = self.trans8_5(x8,t,Y)  
        x8_5 = self.rect8_5(x8_5)

        x9 = (r*x8_5 + (1-r)*x2_5)/2
        x9 = self.trans9(x9,t,Y)  
        x9 = self.rect9(x9)
        x9_5 = self.trans9_5(x9,t,Y)  
        x9_5 = self.rect9_5(x9_5)
        
        x = self.trans10(x9_5,t,Y)  #

        return x
    
    def forward(self, x, t,Y):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        result = self.unet_forwad(x, t,Y)
        return result


class SPDTransform(nn.Module):

    def __init__(self, input_size , output_size, time_dim,Y_dim):
        super(SPDTransform, self).__init__()
        self.increase_dim = None
        input_size = int(input_size)
        output_size = int(output_size)
        time_dim = int(time_dim)

        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(torch.DoubleTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

        self.emb_layer = nn.Sequential( 
            nn.Linear(time_dim,output_size*output_size).double(),
        )

        self.Y_layer = nn.Sequential( 
            nn.Linear(Y_dim,output_size*output_size).double(),
        )

    def forward(self, input,t,Y):
        output = input
        emb = self.emb_layer(t)   
        if self.increase_dim:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1,2), torch.bmm(output, weight))
        m = output.shape[1]
        n = output.shape[0]
        t_emb = emb.reshape(n,m,m)
        output = torch.matmul(torch.matmul(t_emb,output),t_emb.transpose(1,2))
        
        return output

class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
    
class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        #epsilon = -0.01
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]))

    def forward(self, input):
        output = SPDRectifiedFunction.apply(input, self.epsilon)
        return output


class SPDRectifiedFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            u, s, v = x.svd()
            s[s < epsilon[0]] = epsilon[0]
            output[k] = u.mm(s.diag().mm(u.t()))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1); eye = eye.diag()
            grad_input = input.new(input.size(0), input.size(1), input.size(2))
            for k, g in enumerate(grad_output):
                if len(g.shape) == 1:
                    continue

                g = symmetric(g)    

                x = input[k]
                u, s, v = x.svd()
                
                max_mask = s > epsilon
                s_max_diag = s.clone(); s_max_diag[~max_mask] = epsilon; s_max_diag = s_max_diag.diag()
                Q = max_mask.diag().double()
                
                dLdV = 2*(g.mm(u.mm(s_max_diag)))
                dLdS = eye * (Q.mm(u.t().mm(g.mm(u))))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0

                grad_input[k] = u.mm(symmetric(P.t() * u.t().mm(dLdV))+dLdS).mm(u.t())
            
        return grad_input, None


def symmetric(A):
    return 0.5 * (A + A.t())


class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = np.asarray([0] * input_size + [1] * (output_size-input_size), dtype=np.float64)
        self.register_buffer('add', torch.from_numpy(np.diag(add)))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1).double()
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)

        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1,2)))  

        return output


class SPDTransform1(nn.Module):

    def __init__(self, input_size , output_size, time_dim,Y_dim):
        super(SPDTransform1, self).__init__()
        self.increase_dim = None
        input_size = int(input_size)
        output_size = int(output_size)
        time_dim = int(time_dim)

        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(torch.DoubleTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

        self.emb_layer = nn.Sequential( 
            nn.Linear(time_dim,time_dim).double(),
            nn.SiLU().double(), 
            nn.Linear(time_dim,output_size*2).double(),
        )

        self.Y_layer = nn.Sequential( 
            nn.Linear(Y_dim,output_size*output_size).double(),
            nn.SiLU().double(), 
            nn.Linear(output_size*output_size,output_size*output_size).double(),
        )

    def forward(self, input,t,Y):
        output = input
        #emb = self.emb_layer(t)   
        emb_Y = self.Y_layer(Y.double())
        if self.increase_dim:
            output = self.increase_dim(output)
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1,2), torch.bmm(output, weight))
        m = output.shape[1]
        n = output.shape[0]
        
        emb_Y = emb_Y.reshape(n,m,m)
        output = torch.matmul(torch.matmul(emb_Y,output),emb_Y.transpose(1,2))
        
        return output