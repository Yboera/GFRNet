import torch
import os
from datetime import datetime
from code.GFRNet import build_net
from torch.utils.data import DataLoader
from dataset.data_loader import ALLDataset
import cv2
import torch.nn.functional as F

#set path
model_root = ' '
data_root  = ' '
save_root  = ' '

#load model 
def test(pretrain_model, my_model, dataset, save_sal):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    my_model = my_model.to(device)
    my_model.load_state_dict(torch.load(pretrain_model))
    my_model.eval()

    if not os.path.exists(save_sal):
        os.makedirs(save_sal)

    with torch.no_grad():  
        prev_time = datetime.now()
        print(prev_time)
        #for index, (allfocus, gt, fs, depth, names) in enumerate(dataset):
        for index, (allfocus,depth,fs,names) in enumerate(dataset):
            allfocus = allfocus.to(device)
            depth = depth.to(device)
            focal = torch.cat((fs[0],fs[1],fs[2],fs[3],fs[4],fs[5],fs[6],fs[7],fs[8],fs[9],fs[10],fs[11]), dim=0)
            focal = focal.to(device)
            
            # inputs = torch.cat((focal, allfocus, depth), dim=0)
            layer, co_r, co_d = my_model(focal, allfocus, depth)  
            name = names[0]
            print(name)

            res = layer[0]
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res * 255
            cv2.imwrite(save_sal + name.split('.')[0]+'.png', res)
        cur_time = datetime.now()
        print(cur_time)

if __name__ == '__main__':
    my_model = build_net()
    datas = ['DUTLF', 'LFSD', 'HFUT', 'Lytro']
    for data in datas:
        save_sal = save_root + data + '/' 
        test_dataloader = DataLoader(ALLDataset(location=data_root+data+'/', train=False, dataEnhance=True),batch_size=1, shuffle=False)
        test(model_root, my_model, test_dataloader, save_sal)

