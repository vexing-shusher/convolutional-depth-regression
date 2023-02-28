from cdr import *
from cdr_dataset import *

import argparse

import glob
import os

import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from torchvision import transforms

import pandas as pd

from tqdm import tqdm

import logging


def preprocess_idxs(cls_array : list, 
                    logger,
                    maj_class : int = 2, 
                    ovs : float = 2., 
                    ssize : int = 5000,
                    exclude_cls: list = [],
                    ):

    full_maj_idxs = [idx for idx in range(len(cls_array)) if cls_array[idx] == maj_class]
    maj_idxs = []

    #downsampling
    if len(full_maj_idxs) == ssize:
        logger.info("No downsampling required.")
        maj_idxs = full_maj_idxs
    else:
        logger.info(f"Downsampling the class {maj_class}...")
        pbar = tqdm(range(ssize))
        while len(maj_idxs) < ssize:
            idx = np.random.choice(full_maj_idxs)
            if idx not in maj_idxs:
                maj_idxs.append(idx)
                pbar.update(1)
        pbar.close()

    min_classes = [c for c in np.unique(cls_array) if (c != maj_class) and (c not in exclude_cls)]

    #oversampling
    min_idxs = []
    for c in min_classes:
        logger.info(f"Oversampling the class {c}...")
        cur_class_idxs = [idx for idx in range(len(cls_array)) if cls_array[idx] == c]
        tmp = [idx for idx in range(len(cls_array)) if cls_array[idx] == c]
        pbar = tqdm(range(np.ceil(ovs*len(tmp)).astype(np.int32)))
        pbar.update(len(tmp))
        while len(cur_class_idxs) < np.ceil(ovs*len(tmp)):
            cur_class_idxs.append(np.random.choice(tmp))
            pbar.update(1)
        pbar.close()

        min_idxs += cur_class_idxs

    return np.asarray(maj_idxs + min_idxs)

def run_experiment_pipeline(args):

    logger = logging.getLogger("Experiment log")
    logger.setLevel(logging.INFO)
    # Format for our loglines
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Setup console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    beta_key = str(args.beta).replace('.','')

    if args.ds_strategy == "RND":
        ds_key = "full_"
        size_key = ""
    elif args.ds_strategy == "NM2" or args.ds_strategy == "NM1":
        ds_key = args.ds_strategy + "_"
        size_key = f"_s{args.ssize}"

    filename = f"{ds_key}train_data_b{beta_key}{size_key}.pkl"

    if not os.path.exists(f"{args.home_dir}/model_checkpoints/"):
            os.mkdir(f"{args.home_dir}/model_checkpoints/")

    if not os.path.exists(f"{args.home_dir}/model_outputs/"):
        os.mkdir(f"{args.home_dir}/model_outputs/")

    cur_rep = 0
    for checkpoint_name in os.listdir(f"{args.home_dir}/model_checkpoints/"):
        if  checkpoint_name[0] != '.':
            cur_rep = np.max([cur_rep, int(checkpoint_name.split('_')[-1].split('.')[0])]).astype(np.int32)

    cur_rep += 1

    current_paths = []
    
    special_keys = ["fx", "fy", "cls_w"]

    args.config = {
          'bs':512,
          'lr':0.0005,
          'rfactor':2.,
          'cfactor':args.cfactor,
          'n_epochs':300,
          'patience':10,
          'subset_size':args.ssize,
          'oversampling':args.ovs,
          'network':{
              'lin_params':[(512,256),(256,128),(128,64),(64,32),(32,4)],
              'loss':args.loss,
              'delta':1.,
              'decoder':args.decoder,
          },
        }

    square_side = np.min((np.ceil(1242*args.config["cfactor"]/args.config["rfactor"]),np.ceil(375*args.config["cfactor"]/args.config["rfactor"])))

    args.input_shape = (square_side, square_side)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if bool(args.train):

        for n, run in enumerate(range(5)):
            
            exp_number = cur_rep + n
            logger.info(f"Experiment number {exp_number}")

            with open(f"{args.meta_dir}{filename}", 'rb') as f:
                data_json = pickle.load(f)

            skf = StratifiedKFold(n_splits=5)

            random_idxs = preprocess_idxs(data_json["cls"],
                                          logger,
                                          maj_class = 2,
                                          ovs = args.ovs,
                                          ssize = args.ssize)

            cls = np.asarray(data_json["cls"])[random_idxs]

            fold_num = 0

            fn = -1
            for train_idx, test_idx in skf.split(random_idxs, cls):
                fn+=1
                if fn == fold_num:
                    random_train_idxs, random_val_idxs = random_idxs[train_idx], random_idxs[test_idx]

            args.run_num = exp_number

            model_path = f"{args.home_dir}/model_checkpoints/distreg_{args.run_num}.pt"
            current_paths.append(model_path)

            logger.info(f"Running on device: {device}")

            train_data_json = {key:np.asarray(data_json[key])[random_train_idxs] for key in data_json if key not in special_keys}

            logger.info(f"Number of training samples: {len(train_data_json['cls'])}")

            val_data_json = {key:np.asarray(data_json[key])[random_val_idxs] for key in data_json if key not in special_keys}

            for key in special_keys:
                train_data_json[key] = data_json[key]
                val_data_json[key] = data_json[key]   

            transform = transforms.RandomApply([ 
                                          transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                          transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
                                          transforms.RandomAdjustSharpness(0.7, p=0.5),
                                          ])

            train_ds = CDRDataset(data_json=train_data_json, 
                                  home_dir = args.data_dir,
                                    resize_factor=args.config['rfactor'], 
                                    crop_factor=args.config['cfactor'],
                                    maj_class = 2, 
                                    transform=transform)

            valid_ds = CDRDataset(data_json=val_data_json, 
                                  home_dir = args.data_dir,
                                  resize_factor=args.config['rfactor'], 
                                  crop_factor=args.config['cfactor'],)

            train_loader = DataLoader(train_ds, batch_size = args.config['bs'], shuffle=True, pin_memory = False)
            valid_loader = DataLoader(valid_ds, batch_size = 1, shuffle=True, pin_memory = False)

            model = CDR5Model(args)
            model.to(model.device)

            model.fit(train_loader, valid_loader, lr = args.config['lr'], n_epochs = args.config['n_epochs'], patience = args.config['patience'])
            
            del train_ds
            del valid_ds
            del train_loader
            del valid_loader
            del model
            
            torch.cuda.empty_cache()

    if bool(args.eval):

        if not bool(args.train):
            #evaluating the last five checkpoints
            current_paths = [f"{args.home_dir}/model_checkpoints/{item}" for item in os.listdir(f"{args.home_dir}/model_checkpoints/")[-5:]]

        #subsampling test data
        with open(f"{args.meta_dir}val_data.pkl", 'rb') as f:
            test_data_json_1 = pickle.load(f)
        with open(f"{args.meta_dir}test_data.pkl", 'rb') as f:
            test_data_json_2 = pickle.load(f)

        test_data_json = {key:test_data_json_1[key]+test_data_json_2[key] for key in test_data_json_1 if key not in special_keys}

        for key in special_keys:
            test_data_json[key] = test_data_json_1[key]

        test_ds = CDRDataset(data_json=test_data_json,
                             home_dir = args.data_dir, 
                             resize_factor=args.config['rfactor'], 
                             crop_factor=args.config['cfactor'])

        test_loader = DataLoader(test_ds, batch_size = 1, shuffle=False, pin_memory = False)

        output = {}
        
        logger.info("Running evaluation...")
        for run, model_path in enumerate(current_paths):
            fitted_model = torch.load(model_path).to(device)
            fitted_model.eval()

            pred_dict = {}
            true_dict = {}
            bl_dict = {}

            for key in np.unique(test_data_json["cls"]):
                pred_dict[str(key)] = []
                true_dict[str(key)] = []
                bl_dict[str(key)] = []

                if run == 0:
                    output[str(key)] = []

            test_data_json["pred"] = []
            for cls, batch in zip(test_data_json["cls"],test_loader):
                y_, _= fitted_model(batch[0].to(device), batch[2], batch[3], batch[4].to(device))
                pred_dict[str(cls)].append(y_.item())
                test_data_json["pred"].append(y_.item())

            for cls, gtd, bl in zip(test_data_json["cls"], test_data_json["gtd"], test_data_json["md2"]):
                true_dict[str(cls)].append(gtd)
                bl_dict[str(cls)].append(bl)

            class_lengths = [len(true_dict[str(cls)]) for cls in np.unique(test_data_json["cls"])]
            total_len = np.sum(class_lengths)

            #sort by class
            class_mae_pred_vals = [
                mean_absolute_error(true_dict[key], pred_dict[key]) 
                for key in true_dict
            ]

            class_mae_bl_vals = [
                mean_absolute_error(true_dict[key], bl_dict[key]) 
                for key in true_dict
            ]

            for key, val in zip(np.unique(test_data_json["cls"]), class_mae_pred_vals):
                output[f"{key}"].append(val)
                
            avg_pred = np.sum(
                [
                    cls_w * mae_val / total_len 
                    for cls_w, mae_val in zip(class_lengths, class_mae_pred_vals)
                ]
            )
            
            avg_bl = np.sum(
                [
                    cls_w * mae_val / total_len 
                    for cls_w, mae_val in zip(class_lengths, class_mae_bl_vals)
                ]
            )


            distance_bins = [(10*k,10*(k+1)) for k in range(0,7)]

            full_distance_dict = {key1:{key2:[] for key2 in distance_bins} for key1 in ["true", "bl", "pred"]}

            if run == 0:
                for key2 in distance_bins:
                    output[f"{key2}"] = [] 

            for key1, data_dict in zip(["true","bl","pred"], [true_dict, bl_dict, pred_dict]):
                for key2 in data_dict:
                    for item, base in zip(data_dict[key2],true_dict[key2]):
                        for k in range(len(distance_bins)):
                            if base >= distance_bins[k][0] and base < distance_bins[k][1]:
                                full_distance_dict[key1][distance_bins[k]].append(item)

            #calculate distance MAEs
            dist_mae_pred_vals = [
                mean_absolute_error(full_distance_dict['true'][key], full_distance_dict['pred'][key]) 
                for key in full_distance_dict['true']
            ]

            dist_mae_bl_vals = [
                mean_absolute_error(full_distance_dict['true'][key], full_distance_dict['bl'][key]) 
                for key in full_distance_dict['true']
            ]

            for key, val in zip(distance_bins, dist_mae_pred_vals):
                output[f"{key}"].append(val)
                
            if run == 0:
                output["AVG"] = []
            output["AVG"].append(avg_pred)
        
        for key, val in zip(np.unique(test_data_json["cls"]), class_mae_bl_vals):
                output[f"{key}"].append(val)

        for key, val in zip(distance_bins, dist_mae_bl_vals):
                output[f"{key}"].append(val)
                
        output["AVG"].append(avg_bl)

        frame_index = [f"run_{r}" for r in range(1,6)] + ["baseline"]
        frame = pd.DataFrame(data=output, index = frame_index)
        
        cf_str = str(args.cfactor).replace('.','')
        bt_str = str(args.beta).replace('.','')
        
        output_name = f"{cf_str}_{args.decoder}_{args.loss}_{args.ds_strategy}_{args.ssize}_{bt_str}.csv"
        with open(f"{args.home_dir}/model_outputs/{output_name}", 'w') as f:
            frame.to_csv(f, encoding='utf-8', sep=';')
            
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--home_dir", type=str, default="./")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--meta_dir", type=str, default="./ground_truth/full_datasets/")
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--ssize", type=int, default=7000)
    parser.add_argument("--ovs", type=int, default=2)
    parser.add_argument("--cfactor", type=float, default=0.3)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--train", type=int, default=1)
    parser.add_argument("--eval", type=int, default=1)
    parser.add_argument("--ds_strategy", type=str, default="RND")
    parser.add_argument("--decoder", type=str, default="base")
    
    args = parser.parse_args()
    
    run_experiment_pipeline(args)
    
