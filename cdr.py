import torch
from torch.nn import Conv2d, Dropout, MaxPool2d, Linear, ReLU, MSELoss, Module, HuberLoss, LayerNorm, Sequential, Flatten
from torch.nn import Parameter as P
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.models import convnext_tiny, convnext_small
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights

from typing import OrderedDict

import numpy as np

import os

from tqdm import tqdm

class MSEDepthLoss(Module):
    def __init__(self, dmax : float = 80., device=torch.device("cpu")):
        super(MSEDepthLoss, self).__init__()
        self.dmax = dmax
        self.device = device
        
    def forward(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor, cls : torch.Tensor, cls_w : torch.Tensor):
        return MSELoss()(d/self.dmax, gtd/self.dmax)

class RLMSEDepthLoss(Module):
    def __init__(self, dmax : float = 80., device=torch.device("cpu")):
        super(RLMSEDepthLoss, self).__init__()
        self.dmax = dmax
        self.device = device
        
    def forward(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor, cls : torch.Tensor, cls_w : torch.Tensor):
        return torch.sqrt(MSELoss()(torch.log(d+1), torch.log(gtd+1)))
        

class KLDepthLoss(Module):
    
    def __init__(self, dmax : float = 80., device = torch.device("cpu")):
        super(KLDepthLoss, self).__init__()
        
        '''
        Variance-adjusted regression loss, as described in He, Yihui, et al. "Bounding box regression 
        with uncertainty for accurate object detection." 
        Proceedings of the ieee/cvf conference on computer vision and pattern recognition. 2019.
        '''
        
        self.dmax = dmax
        self.device = device
        
    def H1(self, d : torch.Tensor, gtd : torch.Tensor)->torch.Tensor:
        #equal to 1 when |d - gtd|/dmax > 1, else 0
        with torch.no_grad():
            h1 = torch.heaviside((torch.abs(d - gtd)/self.dmax)-1, torch.zeros(1).to(self.device))
        return h1
    
    def H2(self, d : torch.Tensor, gtd : torch.Tensor)->torch.Tensor:
        #equal to 1 when |d - gtd|/dmax < 1, else 0
        with torch.no_grad():
            h2 = 1 - self.H1(d, gtd)
        return h2
    
    def SL1(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor)->torch.Tensor:
        
        return torch.exp(-alpha) * (torch.abs(d - gtd)/self.dmax - 0.5) + 0.5*alpha
    
    def KLD(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor)->torch.Tensor:
        
        return 0.5*torch.exp(-alpha) * ((d-gtd)/self.dmax)**2 + 0.5*alpha
        
    def forward(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor, cls : torch.Tensor, cls_w : torch.Tensor):
        
        #reduction over batch -- mean
        
        return torch.mean(self.H1(d, gtd) * self.SL1(d, gtd, alpha) + self.H2(d, gtd) * self.KLD(d, gtd, alpha))
    
    
class WeightedKLDepthLoss(KLDepthLoss):
    
    def __init__(self, dmax : float = 80., n_classes : int = 7, device = torch.device("cpu")):
        super(WeightedKLDepthLoss, self).__init__(dmax, device)
        
        self.n_classes = n_classes
        
    def forward(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor, cls : torch.Tensor, cls_w : torch.Tensor):
        
        classes_one_hot = F.one_hot(cls.long(), self.n_classes) #shape -- (batch_size, n_classes)
        
        W = classes_one_hot.float() @ cls_w[0].unsqueeze(1) #shape -- (batch_size, n_classes) @ (n_classes, 1) = (batch_size, 1)
        W = W.to(self.device).squeeze()
        W.requires_grad = False
        
        #reduction over batch -- mean
        
        return torch.mean(W*(self.H1(d, gtd) * self.SL1(d, gtd, alpha) + self.H2(d, gtd) * self.KLD(d, gtd, alpha)))
    
class WeightedMSEDepthLoss(MSEDepthLoss):
    
    def __init__(self, dmax : float = 80., n_classes : int = 7, device = torch.device("cpu")):
        super(WeightedMSEDepthLoss, self).__init__(dmax, device)
        
        self.n_classes = n_classes
        
    def forward(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor, cls : torch.Tensor, cls_w : torch.Tensor):
        
        classes_one_hot = F.one_hot(cls.long(), self.n_classes) #shape -- (batch_size, n_classes)
        
        W = classes_one_hot.float() @ cls_w[0].unsqueeze(1) #shape -- (batch_size, n_classes) @ (n_classes, 1) = (batch_size, 1)
        W = W.to(self.device).squeeze()
        W.requires_grad = False
        
        #reduction over batch -- mean
        
        return torch.mean(W*MSELoss(reduction='none')(d/self.dmax, gtd/self.dmax))

class WeightedRLMSEDepthLoss(RLMSEDepthLoss):
    
    def __init__(self, dmax : float = 80., n_classes : int = 7, device = torch.device("cpu")):
        super(WeightedRLMSEDepthLoss, self).__init__(dmax, device)
        
        self.n_classes = n_classes
        
    def forward(self, d : torch.Tensor, gtd : torch.Tensor, alpha : torch.Tensor, cls : torch.Tensor, cls_w : torch.Tensor):
        
        classes_one_hot = F.one_hot(cls.long(), self.n_classes) #shape -- (batch_size, n_classes)
        
        W = classes_one_hot.float() @ cls_w[0].unsqueeze(1) #shape -- (batch_size, n_classes) @ (n_classes, 1) = (batch_size, 1)
        W = W.to(self.device).squeeze()
        W.requires_grad = False
        
        #reduction over batch -- mean
        
        return torch.mean(W*torch.sqrt(MSELoss(reduction='none')(torch.log(d+1), torch.log(gtd+1))))
                
class CDR5Model(Module):
    
    def __init__(self, args):
        super(CDR5Model, self).__init__()

        self.home_dir = args.home_dir
        self.device = args.device
        self.lr = args.config["lr"]
        self.input_shape = args.input_shape
        self.run_num = args.run_num
        
        #list of model parameters
        self.parameters = []

        #feature extractor
        self.backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
            
        #regression block
        lin_out = int(args.config["network"]["lin_params"][0][0])
        
        regression_block = [
            LayerNorm((768,1,1), eps=1e-06, elementwise_affine=True),
            Flatten(start_dim=1, end_dim=-1),
            Linear(768, lin_out),
        ]
        
        linear_block = []

        for k in range(len(args.config["network"]["lin_params"])-1):
            lin_params = args.config["network"]["lin_params"][k]
            linear_block.append(Linear(*(int(p) for p in lin_params)))

        linear_activations = []
        for i in range(len(linear_block)):
            linear_activations.append(ReLU())
            
        for lin, act in zip(linear_block, linear_activations):
            regression_block.append(act)
            regression_block.append(lin)
            
        regression_block.append(ReLU())
            
        self.backbone.classifier = Sequential(
            OrderedDict(
                [
                    (f'{i}',regression_block[i]) 
                    for i in range(len(regression_block))
                ]
            )
        )

        #initialize decoder
        if args.config["network"]["decoder"] == "base":
            self.decoder = self.decoder_base
            out_params = args.config["network"]["lin_params"][-1]
        else:
            self.decoder = self.decoder_dummy
            out_params = (args.config["network"]["lin_params"][-1][0], 1)

        self.out_lin = Linear(*out_params)
        
        sigma_dims = (self.out_lin.weight.shape[-1],1)
        self.sigma_lin = Linear(*sigma_dims, bias=False)
        self.sigma_lin.weight = P(torch.normal(0., 1e-4, tuple(reversed(sigma_dims))))
        
        self.sigma_lin.to(self.device) 
        self.out_lin.to(self.device)
        self.backbone.to(self.device)
        
        #freeze feature extraction layers
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        self.parameters.append({'params':self.backbone.classifier.parameters()})
        self.parameters.append({'params':self.sigma_lin.parameters()})
        self.parameters.append({'params':self.out_lin.parameters()})

        #initialize optimizer
        self.optimizer = Adam(self.parameters, lr=self.lr)

        #initialize loss function
        if args.config["network"]["loss"] == "MSE":
            self.loss = WeightedMSEDepthLoss(dmax=80., n_classes=7, device=self.device)
        elif args.config["network"]["loss"] == "WKLD":
            self.loss = WeightedKLDepthLoss(dmax=80.,n_classes=7, device=self.device)
        #elif args.config["network"]["loss"] == "WRLMSE":
        #    self.loss = WeightedRLMSEDepthLoss(dmax=80.,n_classes=7,device=self.device)  

        
    def decoder_base(self, x : torch.Tensor, fx : float, fy : float, box : torch.Tensor):

        return 0.5 * (fx * x[:,0] / box[:, 2] / (1+x[:,1]) + fy * x[:,2] / box[:,3] / (1+x[:,3]))

    def decoder_dummy(self, x : torch.Tensor, fx : float, fy : float, box : torch.Tensor):

        return x


    def forward(self, x : torch.Tensor, fx : float, fy : float, box : torch.Tensor):

        x = self.backbone(x)
        alpha = self.sigma_lin(x)
        x = self.out_lin(x)

        x = self.decoder(x, float(fx[0]), float(fy[0]), box).to(self.device)

        return x.squeeze(), alpha.squeeze()

    def fit(self, train_loader : DataLoader, valid_loader : DataLoader, lr = 0.001, n_epochs = 100, patience = 10):

        if not os.path.exists(f"{self.home_dir}/model_checkpoints/"):
            os.mkdir(f"{self.home_dir}/model_checkpoints/")

        max_valid_loss = np.inf

        pbar = tqdm(range(n_epochs))

        counter = patience

        for e in pbar:
            
            if counter == 0:
                print(f"\nBest validation loss: {max_valid_loss.item()}")
                break

            epoch_train_loss = 0
            for batch in train_loader:

                self.optimizer.zero_grad()
                #batch[0] -- sample, batch[2] -- fx, batch[3] -- fy, batch[4] -- bbox
                y_pred, alpha = self(batch[0].to(self.device), batch[2], batch[3], batch[4].to(self.device))
                
                cls, cls_w = batch[5].to(self.device), batch[6].to(self.device)

                train_loss = self.loss(y_pred, batch[1].squeeze().to(self.device), alpha, cls, cls_w)
                train_loss.backward()
                self.optimizer.step()
                epoch_train_loss+=train_loss

                torch.cuda.empty_cache()

            epoch_train_loss /= len(train_loader)

            epoch_valid_loss = 0
            
            self.eval()
            for batch in valid_loader:

                y_pred, alpha = self(batch[0].to(self.device), batch[2], batch[3], batch[4].to(self.device))
                cls, cls_w = batch[5].to(self.device), batch[6].to(self.device)
                valid_loss = self.loss(y_pred, batch[1].squeeze().to(self.device), alpha, cls, cls_w)
                epoch_valid_loss+=valid_loss

                torch.cuda.empty_cache()

            self.train()
            epoch_valid_loss /= len(valid_loader)

            pbar.set_postfix({'tr_loss': epoch_train_loss.item(), 'vl_loss':epoch_valid_loss.item()})

            if epoch_valid_loss < max_valid_loss:
                max_valid_loss = epoch_valid_loss
                torch.save(self, f"{self.home_dir}/model_checkpoints/distreg_{self.run_num}.pt")
                counter = patience
            else:
                counter -= 1