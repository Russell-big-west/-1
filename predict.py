import torch
import numpy as np
import cv2
from torch.utils.data._utils.collate import default_collate
import os

from processor.mmt import get_mmt_model
from processor.dataProcessorMMT import get_data_collection


def move_to_device(obj, device):
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')

def inpainting(model, img_path, mask_path, save_fname):
    device = torch.device('cpu')
    batch = get_data_collection(img_path, mask_path)
    batch = move_to_device(default_collate([batch]), device)
    batch['mask'] = (batch['mask'] > 0) * 1
    batch = model(batch)

    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    print(cur_res.shape)

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite('./tmp/draw/{}'.format(save_fname), cur_res):
        raise Exception('保存图片时出错.Error saving thepicture.')
    
    
if __name__ == '__main__':
    device = torch.device('cpu')
    model = get_mmt_model()
    batch = get_data_collection('Places365_val_00000001.png', 'Places365_val_00000001_mask.png')
    batch = move_to_device(default_collate([batch]), device)
    batch['mask'] = (batch['mask'] > 0) * 1
    batch = model(batch)

    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    print(cur_res.shape)

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite('test_out.jpg', cur_res)
