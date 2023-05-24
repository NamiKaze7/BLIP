import os

from models.med import BertConfig
from models.nlvr_encoder import BertModel
from models.vit import interpolate_pos_embed
from models.blip import create_vit, init_tokenizer, is_url

from timm.models.hub import download_cached_file

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, AutoModel, AutoTokenizer
import numpy as np


class BLIP_CLS(nn.Module):
    def __init__(self,
                 bert_dir='/Users/zhz/work/model/bert-base-chinese/',
                 image_size=480,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 num_tags=2
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer,
                                                       drop_path_rate=0.1)
        self.bert = AutoModel.from_pretrained(bert_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)

        self.cls_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.bert.config.hidden_size, num_tags)
        )

    def forward(self, image, text, targets, train=True):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text = self.tokenizer(text, padding='longest', return_tensors="pt").to(image.device)

        output = self.bert(text.input_ids,
                           attention_mask=text.attention_mask,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           )
        hidden_state = output.last_hidden_state[:, 0, :]
        prediction = self.cls_head(hidden_state)

        if train:
            loss = F.cross_entropy(prediction, targets)
            return loss
        else:
            return prediction


def blip_cls(pretrained='', **kwargs):
    model = BLIP_CLS(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']

    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],
                                                                   model.visual_encoder)

    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self', 'self0')
            new_key1 = key.replace('self', 'self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            new_key0 = key.replace('dense', 'dense0')
            new_key1 = key.replace('dense', 'dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % url_or_filename)
    return model, msg