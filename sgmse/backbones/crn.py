import torch, scipy, math
from .shared import BackboneRegistry
import torch.nn as nn
import numpy as np
from pytorch_lightning.core.lightning import LightningModule

torch.manual_seed(0)
torch.cuda.manual_seed(0)

EPS = 1e-6

    
 

class SinusoidalPosEmb(LightningModule):
    def __init__(self, emb_dim):
        super().__init__()
        emb = torch.log(torch.tensor(1000))/(emb_dim//2 -1)
        emb = torch.exp(-emb*torch.arange(emb_dim//2))
        self.register_buffer('W', emb)
    def forward(self, x):
        x = x[:, None]*self.W[None]
        x = torch.cat((x.sin(), x.cos()), dim=-1)
        return x


class Sigma(nn.Module):
    def __init__(self, arg):
        super().__init__()
        #self.gp = GaussianProject(arg) 
        self.gp = SinusoidalPosEmb(arg)
        self.gp.freeze()
        self.act = nn.PReLU()
        self.ln1 = nn.Linear(arg, 3*arg)
        self.ln2 = nn.Linear(3*arg, 3*arg)
        self.ln3 = nn.Linear(3*arg, arg)
    
    def forward(self, x):
        x = torch.log(x)
        x = self.gp(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.ln2(x)
        x = self.act(x)
        x = self.ln3(x)
        x = self.act(x)
        return x



class Encoder(nn.Module):
    def __init__(self, arg, activat='silu', emb_dim=64, time_emb=True):
        super().__init__()
        if activat == 'silu':
            activat = nn.SiLU(arg[1])
        else:
            activat = nn.PReLU(arg[1])
        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Linear(emb_dim, arg[1])
        self.encoder = nn.Sequential(
            nn.Conv2d(*arg, bias=False),
            nn.BatchNorm2d(arg[1]),
            activat,
        )
    def forward(self, x, emb=None):
        x = self.encoder[0](x)
        if self.time_emb:
            emb = self.time_embedding(emb)
            emb = emb.unsqueeze(-1).unsqueeze(-1)
            x = x + emb
        x = self.encoder[1:](x)
        return x

class Decoder(nn.Module):
    def __init__(self, arg, activation=True, activat='silu', norm=True, emb_dim=64, time_emb=True, sigma=False, extra=False):
        super().__init__()
        if activat == 'silu':
            activat = nn.SiLU(arg[1])
        elif activat == 'sigmoid':
            activat = nn.Sigmoid()
        elif activat == 'none':
            activat = nn.Identity()
        else:
            activat = nn.PReLU(arg[1])

        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Linear(emb_dim, arg[1])
        self.sigma = sigma
        self.extra = extra
        if sigma:
            self.noisy_emb = nn.Linear(emb_dim, arg[0])
        if extra:
            self.exln = nn.Linear(arg[0]//2, arg[0]//2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(*arg, bias=False),
        )
        if activation:
            if norm:
                self.decoder.add_module('batchnorm', nn.BatchNorm2d(arg[1]))
            self.decoder.add_module('activation', activat)
    def forward(self, x, emb=None, sigma=None, extra=None):
        c = x.shape[1]//2
        if self.sigma:
            sigma = self.noisy_emb(sigma)
            sigma = sigma.unsqueeze(-1).unsqueeze(-1)
            x = x*sigma
        if self.extra:
            x[:,:c]  = x[:, :c] + extra
            
        x = self.decoder[0](x)
        if self.time_emb:
            emb = self.time_embedding(emb)
            emb = emb.unsqueeze(-1).unsqueeze(-1)
            x = x + emb
        if len(self.decoder) - 1:
            x = self.decoder[1:](x) 
        return x

class MiddleBlock(nn.Module):
    def __init__(self, num_rnn_feature, hidden=256 ):
        super().__init__()
        self.lstm = nn.LSTM(num_rnn_feature, hidden, 1, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(hidden, num_rnn_feature)
        self.activation = nn.PReLU()
        self.norm = nn.InstanceNorm1d(200)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], -1, shape[-1])
        x = x.permute(0, 2, 1)
        x, s = self.lstm(x)
        x = self.norm(x)
        x  = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(shape)
       
        return x     

def cosine_beta(time_steps, s=.008):
    x = torch.linspace(0, time_steps, time_steps+1)
    alpha_cumprod = torch.cos((x/time_steps + s)*math.pi*.5/(1 + s))
    alpha_cumprod = alpha_cumprod/alpha_cumprod[0]
    beta = 1 - alpha_cumprod[1:]/alpha_cumprod[:-1]
    return torch.clip(beta, 0, 0.999)



@BackboneRegistry.register("unet")
class CRN(nn.Module):
    def __init__(self, 
                 window_length=510, 
                 emb_dim=64,
                 encoder_args=None, 
                 decoder_args=None,  
                 activation='silu',
                 time_emb=True,
                 sigma=True,
                 extra=False,
                 condition_output=False,
                 feature_norm=False,
                 diffusion_domain='f',
                 prediction_mode='noise', 
                 **kwargs,
                ):
        super().__init__()

        if encoder_args==None:
            encoder_args=[
                [4, 45, [7, 1], [1, 1], [3, 0]],
                [45, 45, [1, 7], [1, 1], [0, 3]],
                [45, 90, [7, 5], [2, 2], [3, 2]],
                [90, 90, [7, 5], [2, 2], [3, 2]],
                [90, 90, [5, 3], [2, 2], [2, 1]],
                [90, 90, [5, 3], [2, 1], [2, 1]],
                [90, 90, [5, 3], [2, 2], [2, 1]],
                [90, 128, [5, 3], [2, 1], [2, 1]],
               ]   

        if decoder_args==None:
            decoder_args=[
                [256, 90, [5, 3], [2, 1], [2, 1]],
                [180, 90, [5, 3], [2, 2], [2, 1]],
                [180, 90, [5, 3], [2, 1], [2, 1]],
                [180, 90, [5, 3], [2, 2], [2, 1]],
                [180, 90, [7, 5], [2, 2], [3,2]],
                [180, 45, [7, 5], [2, 2], [3, 2]],
                [90, 45, [1, 7], [1, 1], [0, 3]],
                [90, 2, [7, 1], [1, 1], [3, 0]],
               ]   
 

        num_rnn_feature = self.calc_rnn_feature(encoder_args, window_length//2 + 1)
        self.feature_norm = feature_norm
        self.diffusion_domain = diffusion_domain
        self.extra = extra
        self.condition_output = condition_output
        self.time_emb = time_emb
        self.prediction_mode = prediction_mode
        self.c_encoder = nn.ModuleList()
        self.g_encoder = nn.ModuleList()
        for arg in encoder_args:
            self.c_encoder.append(Encoder(arg, activation, time_emb=False))
            self.g_encoder.append(Encoder(arg, activation, emb_dim, time_emb=True))

        arg[0], arg[2], arg[3], arg[4] = arg[1], [3, 3],[1, 1], [1, 1]
        #self.c_middle_block = Encoder(arg, activation, time_emb=False)
        #self.g_middle_block = Encoder(arg, activation, emb_dim, time_emb=True)
        self.c_middle_block = MiddleBlock(num_rnn_feature)
        self.g_middle_block = MiddleBlock(num_rnn_feature)

        self.c_decoder = nn.ModuleList()
        self.g_decoder = nn.ModuleList()
        output_layer_num = len(decoder_args) - 1
        for i, arg in enumerate(decoder_args):
            use_activation = True if output_layer_num -i else False
            norm = use_activation
            if output_layer_num - i :
                self.c_decoder.append(Decoder(arg, use_activation, activation, norm,time_emb=False,))
            else:
                self.c_decoder.append(nn.Identity())
            arg[0] = arg[0] + arg[0]//2
            self.g_decoder.append(Decoder(arg, use_activation, activation, norm, emb_dim, time_emb=True, sigma=False, extra=False ))
        

        
        #self.ploss = PerceptualLoss()
        #self.ploss.freeze()

        if self.time_emb:
            #self.time_embedding = GaussianProject(emb_dim)
            self.time_embedding = Sigma(emb_dim)
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.set_defaults(centered=True)
        return parser



        
    def calc_rnn_feature(self, arg, input_length):
        
        for i in range(len(arg)):
            input_length = (input_length -arg[i][2][0] + 2*arg[i][4][0])//arg[i][3][0] + 1
            
        return input_length*arg[i][1]
        
    def data_align(self, x1, x2):
        s1_1, s1_2 = x1.shape[-1], x1.shape[-2]
        s2_1, s2_2 = x2.shape[-1], x2.shape[-2]
        p1_l = (s1_1 - s2_1)//2
        p1_r = (s1_1 - s2_1) - p1_l
        p2_l = (s1_2 - s2_2)//2
        p2_r = (s1_2 - s2_2) - p2_l
        
        return nn.functional.pad(x2, (p1_l, p1_r, p2_l, p2_r))

    def forward(self, c, t):
        cr, ci = c.real, c.imag
        c = torch.cat((cr, ci), dim=1)
        target = c.clone()
        device = c.device
        time_shape = list(c.shape)

        shape = list(c.shape)
        t = t.float().squeeze() 
        if t.dim() == 0:
            t = t.unsqueeze(-1)

        
        c_skips = []
        g_skips = []
        temb = self.time_embedding(t) 
        g = c.clone()
        for c_encoder, g_encoder in zip(self.c_encoder, self.g_encoder):
            c = c_encoder(c)
            c_skips.append(c)
            g = g_encoder(g, temb)
            g_skips.append(g)

        c =  self.c_middle_block(c)
        g =  self.g_middle_block(g )
        
        for i, (c_decoder, g_decoder)  in enumerate(zip(self.c_decoder, self.g_decoder)):
            c_skip = c_skips.pop()
            g_skip = g_skips.pop()
            c = self.data_align(c_skip, c)
            g = self.data_align(g_skip, g)
            g = torch.cat((g, c), dim=1)
            g = g_decoder(torch.cat((g, g_skip), dim=1), temb,)
            if len(self.c_decoder) - i -1 :
                c = c_decoder(torch.cat((c, c_skip), dim=1) )


        g = g[:, 0] + 1j*g[:, 1]
        return g



if __name__ == '__main__':

    model = CRN() 
    print(model)
    x = torch.randn(16, 2, 256, 100)
    t = torch.randint(10, (16,))
    os = model(x, t)
    print(os.shape )

    count = 0
    for n, param in model.named_parameters():
        count += param.numel()
        #if param.grad is None:
        #    print(n)
    print('model parameter(M):', count/1e6)
