import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

import os
import sys
import argparse
import glob

from tqdm import tqdm

import pickle
import PIL.Image as pil

from cdr_dataset import CDRDataset

def NM2_downsampling(subset_size : int,
                     KN : int,
                     data_json : dict,
                     data_dir : str = "../datasets/",
                     data_path : str = "./processed_data/full_datasets/",
                     rfactor : float = 2., 
                     cfactor : float = 0.3,
                     save_intermediate : bool = False,
                    )->dict:

    #subsampling
    car_idxs = [idx for idx in range(len(data_json["paths"])) if data_json["cls"][idx] == 2]
    other_idxs = [idx for idx in range(len(data_json["paths"])) if data_json["cls"][idx] != 2]

    maj_data_json = {key:np.asarray(data_json[key])[car_idxs] for key in data_json if key not in ["fx", "fy"]}
    min_data_json = {key:np.asarray(data_json[key])[other_idxs] for key in data_json if key not in ["fx", "fy"]}

    maj_data_json["fx"] = data_json["fx"]
    maj_data_json["fy"] = data_json["fy"]

    min_data_json["fx"] = data_json["fx"]
    min_data_json["fy"] = data_json["fy"]

    maj_ds = CDRDataset(data_json=maj_data_json, data_dir=data_dir, resize_factor=rfactor, crop_factor=cfactor)
    min_ds = CDRDataset(data_json=min_data_json, data_dir=data_dir, resize_factor=rfactor, crop_factor=cfactor)
    min_imgs = torch.cat([item[0].unsqueeze(0) for item in min_ds], dim=0)
    metric = SSIM()

    maj_to_min_mdists = []
    
    #get average euclidean distance from each car sample to KN samples of other classes
    for maj_sample in tqdm(maj_ds):
        
        maj_batch = maj_sample[0].unsqueeze(0).repeat((len(min_imgs),1,1,1),0)
        
        distances = torch.mean(torch.flatten(metric(maj_batch, min_imgs), start_dim=1),dim=-1)

        #we need distances to the farthest members of the minority classes (near-miss-2)
        max_idx = torch.argsort(-distances)[0:KN]

        mdist = torch.mean(distances[max_idx]).item()
        maj_to_min_mdists.append(mdist)
        

    #get <subset_size> car samples with biggest average distance values
    down_idxs = np.argsort(-np.asarray(maj_to_min_mdists))[:subset_size]

    #compile the downsampled dataset
    down_maj_data_json = {key:list(np.asarray(maj_data_json[key])[down_idxs]) for key in data_json if key not in ["fx", "fy"]}

    #put the minority class back in
    for key in down_maj_data_json:
            down_maj_data_json[key] += list(min_data_json[key])

    down_maj_data_json["fx"] = data_json["fx"]
    down_maj_data_json["fy"] = data_json["fy"]

    if save_intermediate:

        #save the downsampled json
        with open(f"{data_path}/NM2_train_data.pkl", 'wb') as f:
            pickle.dump(down_maj_data_json,f)
            
    return down_maj_data_json
        

def NM1_downsampling(subset_size : int,
                     KN : int,
                     data_json : dict,
                     data_dir : str = "../datasets/",
                     data_path : str = "./processed_data/full_datasets/",
                     rfactor : float = 2., 
                     cfactor : float = 0.3,
                     save_intermediate : bool = False,
                    )->dict:

    #subsampling
    car_idxs = [idx for idx in range(len(data_json["paths"])) if data_json["cls"][idx] == 2]
    other_idxs = [idx for idx in range(len(data_json["paths"])) if data_json["cls"][idx] != 2]

    maj_data_json = {key:np.asarray(data_json[key])[car_idxs] for key in data_json if key not in ["fx", "fy"]}
    min_data_json = {key:np.asarray(data_json[key])[other_idxs] for key in data_json if key not in ["fx", "fy"]}

    maj_data_json["fx"] = data_json["fx"]
    maj_data_json["fy"] = data_json["fy"]

    min_data_json["fx"] = data_json["fx"]
    min_data_json["fy"] = data_json["fy"]

    maj_ds = CDRDataset(data_json=maj_data_json, data_dir=data_dir, resize_factor=rfactor, crop_factor=cfactor)
    min_ds = CDRDataset(data_json=min_data_json, data_dir=data_dir, resize_factor=rfactor, crop_factor=cfactor)
    min_imgs = torch.cat([item[0].unsqueeze(0) for item in min_ds], dim=0)
    metric = SSIM()

    maj_to_min_mdists = []
    
    #get average euclidean distance from each car sample to KN samples of other classes
    for maj_sample in tqdm(maj_ds):
        
        maj_batch = maj_sample[0].unsqueeze(0).repeat((len(min_imgs),1,1,1),0)
        
        distances = torch.mean(torch.flatten(metric(maj_batch, min_imgs), start_dim=1),dim=-1)

        #we need distances to the closest members of the minority classes (near-miss-2)
        max_idx = torch.argsort(distances)[0:KN]

        mdist = torch.mean(distances[max_idx]).item()
        maj_to_min_mdists.append(mdist)
        

    #get <subset_size> car samples with biggest average distance values
    down_idxs = np.argsort(-np.asarray(maj_to_min_mdists))[:subset_size]

    #compile the downsampled dataset
    down_maj_data_json = {key:list(np.asarray(maj_data_json[key])[down_idxs]) for key in data_json if key not in ["fx", "fy"]}

    #put the minority class back in
    for key in down_maj_data_json:
            down_maj_data_json[key] += list(min_data_json[key])

    down_maj_data_json["fx"] = data_json["fx"]
    down_maj_data_json["fy"] = data_json["fy"]

    if save_intermediate:

        #save the downsampled json
        with open(f"{data_path}/NM1_train_data.pkl", 'wb') as f:
            pickle.dump(down_maj_data_json,f)
            
    return down_maj_data_json

def generate_class_weights(data_json : dict, data_json_path : str, beta = 0.99)->None:
    
    unique_classes = np.unique(data_json["cls"])
    
    n_classes = [len(np.asarray(data_json["cls"])[data_json["cls"] == k]) for k in unique_classes]
    
    effective_num = 1 - np.power(beta, np.asarray(n_classes))
    weights = (1-beta)/effective_num
    
    #normalize
    weights = weights / np.sum(weights) * len(unique_classes)
    
    data_json["cls_w"] = weights
    
    with open(data_json_path, 'wb') as f:
        pickle.dump(data_json, f)
        
    return None


def run_monodepth2(data_json : dict,
                   data_json_path : str,
                   data_path : str,
                   model_name : str = "mono+stereo_1024x320",
                   STEREO_SCALE_FACTOR : float = 5.4,
                   margin : float = 2.,
                   save_point_cloud : bool = False,
                   save_intermediate : bool = False,
                  ):
    
    #load the model
    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()
        
    data_json['md2'] = []
    
    if save_point_cloud:
        data_json['md2_point_cloud'] = []
    
    pbar = tqdm(total=len(data_json['paths']))
    
    for image_path, bbox in zip(data_json['paths'], data_json['boxes']):
        
        input_image = pil.open(f"{data_path}{image_path}").convert('RGB')
        original_width, original_height = input_image.size

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        
        with torch.no_grad():
            features = encoder(input_image_pytorch)
            outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        
        disp_resized = torch.nn.functional.interpolate(disp,
    (original_height, original_width), mode="bilinear", align_corners=False)
        
        _, depth = disp_to_depth(disp_resized, 0.1, 100)
        
        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
        
        
        dist = get_gt_distance(gt_depth = metric_depth, 
                                       bbox = bbox, 
                                       margin = margin)
        
        data_json['md2'].append(dist)
        
        if save_point_cloud:
            data_json['md2_point_cloud'].append(metric_depth)
            
        pbar.update(1)
        
    pbar.close()
        
    name = '.' + data_json_path.split('.')[1] + '_md2.pkl'
    
    #Saving data json
    if save_intermediate:
        with open(name, 'wb') as f:
            pickle.dump(data_json, f)
        
    return data_json


def get_image_paths(data_dir : str, split_path : str = "./monodepth2/splits/eigen_full/train_files.txt")->list:

    with open(split_path, 'r') as f:

        lines = f.read().split('\n')

        frame_names = [line.split(' ')[1].rjust(10, '0') + '.jpg' for line in lines]

        image_folders = []

        for line in lines:
                if (line.split(' ')[-1] == 'l'): 
                    image_folders.append("image_02")  
                else: 
                    image_folders.append("image_03")


        full_paths = []

        for line, img_dir, frame_name in zip(lines, image_folders, frame_names):

            if img_dir == "image_02":

                fp = f"{data_dir}{line.split(' ')[0]}/{img_dir}/data/{frame_name}"
                #print(fp)
                #print(data_dir)

                full_paths.append(fp)
    
    return full_paths


def get_gt_distance(gt_depth : np.ndarray, 
                    bbox : np.ndarray, 
                    MIN_DEPTH : float = 1e-3,
                    MAX_DEPTH : float = 80.,
                    margin : float = 2.,
                   )->float:
    
    #get bounding box in array indices
    x0, y0, x1, y1 = bbox[:4]
    
    #preprocess ground truth
    sample = gt_depth.squeeze()[y0:y1, x0:x1] #crop out bounding box
    
    sample[sample > MAX_DEPTH] = MAX_DEPTH #cap depth from above
    
    sample = sample[sample > MIN_DEPTH] #exclude non-significant points
    
    sample_mean = np.mean(sample)
    
    #remove artifacts
    sample = sample[np.logical_and(sample < margin*sample_mean, sample > (1/margin)*sample_mean)]
    
    return np.mean(sample) #return predicted distance

def run_yolo(split_path : str, 
             data_dir : str, 
             gt_path : str,
             out_path : str,
             conf_thresh : float = 0.8, margin : float = 2.,
             save_point_cloud : bool = False,
             save_intermediate : bool = False,
            )->dict:

    data = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    image_paths = get_image_paths(data_dir, split_path)

    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

    relevant_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']

    out = {'paths':[], 'boxes':[], 'gtd':[], 'cls':[]}
    
    if save_point_cloud:
        out["point_cloud"] = []
    
    assert len(image_paths) == len(data), f"Number of image paths: {len(image_paths)}, number of gtd: {len(data)}"
    
    pbar = tqdm(total=len(image_paths))
    
    for image_path, gt_depth in zip(image_paths, data):

        boxes = yolo(image_path).pandas().xyxy[0]

        relevant_idx = [idx for idx in boxes.index if boxes["name"][idx] in relevant_class_names]
        confident_idx = [idx for idx in relevant_idx if boxes["confidence"][idx] > conf_thresh]

        if len(confident_idx) > 0:

            #vals = array(#n_boxes, [xmin, ymin, xmax, ymax, conf, cls])
            vals = np.ceil(boxes.iloc[confident_idx, :-1].to_numpy()).astype(np.int32)

            for bbox in vals:

                dist = get_gt_distance(gt_depth = gt_depth, 
                                       bbox = bbox, 
                                       margin = margin)
                
                splt = image_path.split("/")
                kitti_index = splt.index("KITTI")

                out['paths'].append("/".join(splt[kitti_index:]))
                out['boxes'].append(tuple(bbox[:4]))
                out['gtd'].append(dist)
                out['cls'].append(bbox[-1])
                if save_point_cloud:
                    out['point_cloud'].append(gt_depth)
                
        torch.cuda.empty_cache()
        pbar.update(1)
    
    pbar.close()
    
    name = split_path.split('/')[-1].split('.')[0] + '.pkl'
    
    if save_intermediate:
        with open(f"{out_path}/{name}", 'wb') as f:
            pickle.dump(out, f)
        
    return out, f"{out_path}/{name}"

def split_train_lines(n_parts : int = 6, split_path : str = "./splits/eigen_full/train_files.txt")->None:

    with open(split_path, 'r') as f:

        lines = f.read().split('\n')[:-1]

    lines = [line for line in lines if line[-1] == 'l']

    for n in range(0, n_parts):

        st = ""

        for k in range(int(len(lines)*n/6), int(len(lines)*(n+1)/6)):

            st += lines[k]
            st += '\n'

        name = split_path.split('/')[-1].split('.')[0]+f"_{n}.txt"

        pth = split_path.split('/')[:-1]
        pth.append(name)

        with open('/'.join(pth), 'w', encoding='utf-8') as fp:
            fp.write(st)
            
    return None

def generate_gt_json(conf_thresh : float = 0.8, 
                     margin : float = 2., 
                     data_root : str = "../datasets/",
                     save_intermediate : bool = False,
                    )->list:
    
    gt_dir = "./monodepth2/ground_truth/"
    assert os.path.exists(gt_dir), "Ground truth directory does not exist."
    
    #list the files in the directory
    num_files = len(glob.glob(os.path.join(gt_dir, "*.npz")))
    
    
    postfix = [f"_{k}" for k in range(0,num_files-2)] #accounting for the splitted train files
    postfix += ["", ""] # val and test files do not have number indices

    subsets = ["train" for k in range(0,num_files-2)]
    subsets += ["val", "test"]
    
    results = []
    paths = []
    
    out_path = "./processed_data/"
    data_dir = f"{data_root}/KITTI/"

    for pf, ss in zip(postfix, subsets):
        split_path = f"{gt_dir}/{ss}_files{pf}.txt"
        gt_path = f"{gt_dir}/gt_depths_{ss}{pf}.npz"
        
        if ss in ["val", "test"]:
            save_point_cloud = True
        else:
            save_point_cloud = False

        out, path = run_yolo(split_path, data_dir, gt_path, out_path, conf_thresh, margin, save_point_cloud)
        
        #relabel classes
        new_cls = []
        for c in out["cls"]:
            if c < 4:
                new_cls.append(c)
            else:
                new_cls.append(c-1)
        out["cls"] = new_cls
        
        results.append(out)
        paths.append(path)
        
    return results, paths

def run_data_pipeline(args):
    
    conf_thresh = args.conf_thresh
    margin = args.margin
    save_intermediate = args.save_intermediate
    subset_size = args.subset_size
    KN = args.KN
    betas = args.betas
    fx = args.fx
    fy = args.fy
    data_dir = args.data_dir
    out_path = args.out_path
    
    
    #generate ground truth data
    data_jsons, data_json_paths = generate_gt_json(conf_thresh = conf_thresh, 
                                                   margin = margin, 
                                                   data_root = data_dir,
                                                   save_intermediate = save_intermediate)

    for i, data_json in enumerate(data_jsons):
        assert data_json["paths"][0].split('/')[0] == "KITTI", f"Split {i} is incorrect after stage 1"
    
    data_jsons_md2 = []
    
    for n, data_json, data_json_path in zip(range(len(data_jsons)), data_jsons, data_json_paths):
        
        if n < 6:
            save_point_cloud = False
            data_jsons_md2.append(data_json)
        else:
            save_point_cloud = True
        
            data_json_md2 = run_monodepth2(data_json = data_json, 
                                           data_json_path=data_json_path, 
                                           data_path = data_dir,
                                           margin=margin, 
                                           save_point_cloud = save_point_cloud,
                                           save_intermediate = save_intermediate)
        
            data_jsons_md2.append(data_json_md2)
        
    del data_jsons

    for i, data_json in enumerate(data_jsons_md2):
        assert data_json["paths"][0].split('/')[0] == "KITTI", f"Split {i} is incorrect after stage 2"
    
    full_train_data = {key:[] for key in data_jsons_md2[0]}

    for n, data_json in enumerate(data_jsons_md2):
        if n < 6:
            for key in data_json:
                full_train_data[key] += data_json[key]
        elif n == 6:
            full_val_data = data_json
            full_val_data["fx"] = fx
            full_val_data["fy"] = fy
        else:
            full_test_data = data_json
            full_test_data["fx"] = fx
            full_test_data["fy"] = fy
            
    full_train_data["fx"] = fx
    full_train_data["fy"] = fy

    del data_jsons_md2
    
    
    #create NM2 and NM1 downsampled train data
    NM2_train_data = NM2_downsampling(data_json=full_train_data,
                                      subset_size = subset_size,
                                      KN = KN,
                                      data_dir = data_dir,
                                      data_path = out_path,
                                      save_intermediate = save_intermediate,
                                     )
    
    
    NM1_train_data = NM1_downsampling(data_json=full_train_data,
                                      subset_size = subset_size,
                                      KN = KN,
                                      data_dir = data_dir,
                                      data_path = out_path,
                                      save_intermediate = save_intermediate,
                                     )
    
    #generate class weights for the full train data and save the result
    generate_class_weights(data_json = full_train_data, 
                           data_json_path = f"{out_path}/full_train_data_b{str(0.00).replace('.','')}.pkl", 
                           beta=0.0) #no class weighting
    
        #save val and test data
    generate_class_weights(data_json = full_val_data, 
                           data_json_path = f"{out_path}/val_data.pkl", 
                           beta=0.00)
    
    generate_class_weights(data_json = full_test_data, 
                           data_json_path = f"{out_path}/test_data.pkl", 
                           beta=0.00)
    
    for beta in betas:
    
        generate_class_weights(data_json = full_train_data, 
                               data_json_path = f"{out_path}/full_train_data_b{str(beta).replace('.','')}.pkl", 
                               beta=beta)

        generate_class_weights(data_json = NM2_train_data, 
                               data_json_path = f"{out_path}/NM2_train_data_b{str(beta).replace('.','')}_s{subset_size}.pkl", 
                               beta = beta)

        generate_class_weights(data_json = NM1_train_data, 
                               data_json_path = f"{out_path}/NM1_train_data_b{str(beta).replace('.','')}_s{subset_size}.pkl", 
                               beta = beta)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_thresh",type=float,default=0.8)
    parser.add_argument("--margin",type=float,default=2)
    parser.add_argument("--save_intermediate",type=bool,default=False)
    parser.add_argument("--subset_size",type=int,default=7000)
    parser.add_argument("--KN",type=int,default=10)
    parser.add_argument('--betas', nargs='+', type=float, default=0.0)
    parser.add_argument("--fx",type=float,default=0.58)
    parser.add_argument("--fy",type=float,default=1.92)
    parser.add_argument("--data_dir",type=str,default="../datasets/")
    parser.add_argument("--out_path",type=str,default="./processed_data/full_datasets/")
    
    assert os.path.exists("./monodepth2/"), "The monodepth2 folder is missing from the root project folder."
    sys.path.append("./monodepth2/")
    
    #importing required modules from the monodepth2 project
    from monodepth2 import networks
    from monodepth2.utils import download_model_if_doesnt_exist
    from monodepth2.layers import disp_to_depth, SSIM
    
    args = parser.parse_args()
    
    #create the directories for outputs
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    
    run_data_pipeline(args)