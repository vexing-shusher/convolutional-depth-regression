import torch

from torch.utils.data import Dataset

import numpy as np

import pickle

import PIL.Image as pil

from torchvision import transforms


class CDRDataset(Dataset):
    
    def __init__(self, 
                 data_json, 
                 home_dir,
                 maj_class = 2,
                 resize_factor = 1, 
                 crop_factor = 0.1, 
                 size = (1242, 375),
                 transform = None)->tuple:
        
        self.home_dir = home_dir
        self.maj_class = int(maj_class)

        self.gt_distances = np.asarray(data_json['gtd']).astype(np.float32)
        
        #paths = []
        #for item in data_json['paths']:
        #    split = item.split('/')
        #    idx = split.index("KITTI")
        #    paths.append("/".join(split[idx:]))
            
        #self.image_paths = np.asarray(paths)
        
        self.image_paths = np.asarray(data_json['paths'])

        self.resize_factor = resize_factor
        self.crop_factor = crop_factor

        self.size = (int(size[0] / resize_factor), int(size[1] / resize_factor))
        
        self.boxes = torch.as_tensor(data_json['boxes'], dtype=torch.float32)
        self.cls = torch.as_tensor(data_json['cls'])
        if 'cls_w' in data_json:
            self.cls_w = torch.as_tensor(data_json['cls_w'], dtype=torch.float32)
        else:
            self.cls_w = torch.ones(len(np.unique(data_json['cls'])), dtype=torch.float32)
        
        x1 = torch.as_tensor(self.boxes[:,0] / size[0], dtype=torch.float32)
        y1 = torch.as_tensor(self.boxes[:,1] / size[1], dtype=torch.float32)
        
        w = torch.as_tensor((self.boxes[:,2] - self.boxes[:,0]) / size[0], dtype=torch.float32)
        h = torch.as_tensor((self.boxes[:,3] - self.boxes[:,1]) / size[1], dtype=torch.float32)
        
        
        
        self.boxes[:,0] = x1
        self.boxes[:,1] = y1
        self.boxes[:,2] = w
        self.boxes[:,3] = h
        

        self.fx = data_json["fx"]
        self.fy = data_json["fy"]
        

        #pre-calculate square crop windows
        #midpoint of the bounding box
        x0 = torch.ceil((self.boxes[:,0] + self.boxes[:,2] * 0.5) * self.size[-2]).to(torch.int32)
        y0 = torch.ceil((self.boxes[:,1] + self.boxes[:,3] * 0.5) * self.size[-1]).to(torch.int32)

        #1/2 of the side of the square window around the midpoint
        self.half = np.ceil(0.5 * self.crop_factor * np.min(self.size)).astype(int)

        #calculate bounds

        self.x1 = (x0 - self.half).to(torch.int32)
        self.x1[self.x1 < 0] = 0

        self.x2 = (x0 + self.half).to(torch.int32)
        self.x2[self.x2 > self.size[-2]] = self.size[-2]

        self.y1 = (y0 - self.half).to(torch.int32)
        self.y1[self.y1 < 0] = 0

        self.y2 = (y0 + self.half).to(torch.int32)
        self.y2[self.y2 > self.size[-1]] = self.size[-1]

        self.delta_x = self.x2 - self.x1
        self.delta_y = self.y2 - self.y1


        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x : x

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, in_image_path):
        
        img = pil.open(in_image_path).convert('RGB') 
        
        return transforms.ToTensor()(img.resize(self.size))
   
    def __getitem__(self, idx):

        in_image_path = self.home_dir+self.image_paths[idx]

        inp = self.load_image(in_image_path)

        #crop the image and pad
        p1 = np.ceil(self.half - self.delta_x[idx].item()/2).astype(int)
        p2 = np.ceil(self.half - self.delta_y[idx].item()/2).astype(int)
        sample = transforms.functional.pad(inp[:, self.y1[idx]:self.y2[idx], self.x1[idx]:self.x2[idx]], 
                                         padding=[p1,p2])
        
        if self.cls[idx] == self.maj_class:
          sample = sample[:,0:int(self.half*2),0:int(self.half*2)]
        else:
          sample = self.transform(sample[:,0:int(self.half*2),0:int(self.half*2)])

        target = torch.as_tensor(self.gt_distances[idx], dtype=torch.float32)

        return sample, target, self.fx, self.fy, self.boxes[idx], self.cls[idx], self.cls_w