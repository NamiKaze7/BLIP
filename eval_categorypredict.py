'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os

import pandas as pd
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_cls import blip_cls
import utils
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()
    test_df = pd.read_csv(os.path.join(config['ann_root'], 'test.csv')).reset_index()
    category2id = json.load(open(config['c2i_root']))
    id2category = {}
    for k,v in category2id.items():
        id2category[v] = k
    #category2name = json.load(open(config['c2n_root']))
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    ret = []
    idx = 0

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = images.to(device), targets.to(device)
        text = list(text)
        prediction = model(images, text, targets=targets, train=False)

        top_values, top_indices = torch.topk(prediction, k=config['top_k'], dim=1)
        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))

        probs = top_values.detach().cpu().numpy().tolist()
        cate_ids = top_indices.detach().cpu().numpy().tolist()
        for i in range(len(text)):
            item_dict = test_df.loc[idx].to_dict()
            idx += 1
            item_dict.update({'text': text[i], 'pred': pred_class[i].item(),
                              'label': targets[i].item()})
            top_k_prob = probs[i]
            top_k_cate = cate_ids[i]
            for kidx in range(len(top_k_cate)):
                item_dict.update({f'p_category_name{kidx+1}_mm': id2category[top_k_cate[kidx]],
                                  f'probability{kidx+1}': top_k_prob[kidx]})


            ret.append(item_dict)
    return ret


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    test_dataset = create_dataset('category_predict_eval', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([test_dataset], [False], num_tasks, global_rank)
    else:
        samplers = [None]

    test_loader = create_loader([test_dataset], samplers,
                                batch_size=[config['batch_size_test']], num_workers=[4],
                                is_trains=[False], collate_fns=[None])[0]

    category2id = json.load(open(config['c2i_root']))

    #### Model #### 
    print("Creating model")
    model = blip_cls(pretrained=config['pretrained'], image_size=config['image_size'],
                     vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                     num_tags=len(category2id), bert_dir=config['bert_dir'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    test_result = evaluate(model_without_ddp, test_loader, device, config)
    save_result(test_result, args.result_dir, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/cls.yaml')
    parser.add_argument('--output_dir', default='output/eval_cls')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        args.device = 'cpu'
        args.distributed = False
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.result_dir, 'config.yaml'), 'w'))

    main(args, config)
