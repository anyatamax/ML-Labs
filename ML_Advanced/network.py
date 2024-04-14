
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm


class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_conv1 = nn.Conv1d(in_channels=hid_size, out_channels=hid_size * 2, kernel_size=3)
        self.title_relu1 = nn.ReLU()
        self.title_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.title_flat = nn.Flatten()       
        
        self.full_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=hid_size)
        self.full_conv1 = nn.Conv1d(in_channels=hid_size, out_channels=hid_size * 2, kernel_size=3)
        self.full_relu1 = nn.ReLU()
        self.full_conv2 = nn.Conv1d(in_channels=hid_size * 2, out_channels=hid_size * 2, kernel_size=3)
        self.full_relu2 = nn.ReLU()
        self.full_bn = nn.BatchNorm1d(hid_size * 2)
        self.full_conv3 = nn.Conv1d(in_channels=hid_size * 2, out_channels=hid_size * 2, kernel_size=3)
        self.full_relu3 = nn.ReLU()
        self.full_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.full_flat = nn.Flatten()
        
        self.category_out = nn.Linear(in_features=n_cat_features, out_features=hid_size * 2)

        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size * 2)
        self.inter_relu = nn.ReLU()
        self.final_dense = nn.Linear(in_features=hid_size * 2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_conv1(title_beg)
        title = self.title_relu1(title)
        title = self.title_pool(title)
        title = self.title_flat(title)
        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_conv1(full_beg)
        full = self.full_relu1(full)
        full = self.full_conv2(full)
        full = self.full_relu2(full)
        full = self.full_bn(full)
        full = self.full_conv3(full)
        full = self.full_relu3(full)
        full = self.full_pool(full)
        full = self.full_flat(full)     
        
        category = self.category_out(input3)      
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.inter_dense(concatenated)
        out = self.inter_relu(out)
        out = self.final_dense(out)

        return out
        
        return out