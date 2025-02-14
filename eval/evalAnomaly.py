# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import torch
import random
from PIL import Image
import numpy as np
import importlib
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, ToTensor, Resize

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="AML_Project_Anomaly_Segmentation/datasets/RoadAnomaly21/images/*.png",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="AML_Project_Anomaly_Segmentation/trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default="msp")
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--void', action='store_true')
    
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = 'AML_Project_Anomaly_Segmentation/eval/' + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    # print ("Loading model: " + modelpath)
    # print ("Loading weights: " + weightspath)

    model_file = importlib.import_module(args.loadModel[:-3])
    model = model_file.Net(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                elif args.loadModel != "erfnet.py":
                    own_state["module."+name].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=False))
    # print ("Model and weights LOADED successfully")
    model.eval()

    import torchvision.transforms as T

    # Preprocessing
    image_transform = Compose([Resize((512, 1024), Image.BILINEAR), ToTensor()])
    target_transform = Compose([Resize((512, 1024), Image.NEAREST)])
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        # print(path)
        images = image_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            result = model(images)
        if args.loadModel == "bisenet.py":
            result = result[1]

        if args.void:
            anomaly_result = -result[:, 19, :, :].cpu().numpy().squeeze()
        elif args.method == 'msp':
            result /= args.temp
            softmax_probs = torch.nn.functional.softmax(result, dim=1) # Softmax sulle predizioni del modello
            msp = torch.max(softmax_probs, dim=1)[0].cpu().numpy().squeeze() # Calcolo MSP
            anomaly_result = 1.0 - msp # Anomaly score basato su MSP
        elif args.method == 'ml':
            max_logit, _ = torch.max(result, dim=1) # computing max logit for each pixel
            anomaly_result = -max_logit.cpu().numpy().squeeze() # lower logits to higher anomaly 
        elif args.method == 'me':
            probs = torch.nn.functional.softmax(result, dim=1)
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=1)
            anomaly_result = entropy.data.cpu().numpy().squeeze()

        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "FS_Static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(target_transform(mask))

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    # print("val_label:", val_label)
    # print("val_out:", val_out)

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()
