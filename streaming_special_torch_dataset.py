#!usr/bin/env python
#
# Copyright 2017 houjingyong@gmail.com
# 
# MIT Lisence

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function    

import numpy as np
import torch
from torch.utils.data import Dataset

import kaldi_io
import htk_io
import raw_io

FunctionDict = {'htk_reader': htk_io.htk_read,
                'htk_label_reader': htk_io.htk_read,
                'kaldi_reader': kaldi_io.read_mat,
                'kaldi_label_reader': kaldi_io.read_vec_int,
                'raw_list_reader':  raw_io.read_list
                };

def get_fn_list(fn_name_list):
    fn_list = [FunctionDict[x] for x in fn_name_list]
    return fn_list

def splice_feats(feat, left, right):
    """Splice feature. We first pad each utterance lc frames left and rc frames right.
    Then we use a sliding window to select lc+rc matrices, and concatenate them.

    Args:
        feat (numpy.ndarray): Input feature of an utterance
        left: Left context for splicing.
        right: Right context for splicng.

    Returns:
        spliced feat (numpy.ndarray)
    """
    if left==0 and right==0:
        return feat
    sfeat = []

    num_row = feat.shape[0]
    f0 = feat[0, :]
    fT = feat[num_row-1, :]
    # Repeat the first frame
    pad0 = np.tile(f0, (left, 1))
    # Repeat the last frame
    padT = np.tile(fT, (right, 1))

    pad_feat = np.concatenate([pad0, feat, padT], 0)

    for i in range(0, left+right+1):
        # Splice feat
        sfeat.append(pad_feat[i:i+num_row,:])

    spliced_feat = np.concatenate(sfeat, 1)

    return spliced_feat

class StreamingTorchDataset(Dataset):
    def __init__(self, meta_file, fn_name_list, left_context, right_context, has_label=True):
        with open(meta_file) as fid:
            self.metadata = [ line.strip().split() for line in fid ]
        self.load_fns = get_fn_list(fn_name_list)
        self.left_context = left_context
        self.right_context = right_context
        self.has_label = has_label


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        items = self.metadata[index]
        utt_id = items[0]
        feat_path = items[1]
        feat = self.load_fns[0](feat_path)
        splice_feat = splice_feats(feat, self.left_context, self.right_context)
        if self.has_label:
            label_path = items[2]
            label = self.load_fns[1](label_path)
            # here label is usually squence level label
            return utt_id, splice_feat, label
        else:
            return utt_id, splice_feat

WINDOW_SIZE=0

def collate_fn(batch):
    """Put each data field into a tensor with outer dimension batch size.

    Args:
        batch: A list of tuple (feat, label) for training or (utt_id, feat) for testing
    """
    # batch is list and batch[i] is tuple
    if len(batch[0]) == 2:
        #testting
        batch.sort(key=lambda x: x[:][1].shape[0], reverse=True)
        keys = []
        lengths = []
        feats_padded = []
        max_len = max((batch[0][1].shape)[0], 99)

        for i in range(len(batch)):
            keys.append(batch[i][0])
            act_len = (batch[i][1].shape)[0]
            pad_len = max_len - act_len
            feats_padded.append(np.pad(batch[i][1], ((WINDOW_SIZE, pad_len), (0, 0)), "constant"))
            lengths.append(act_len)
        return keys, torch.from_numpy(np.array(lengths)), torch.from_numpy(np.array(feats_padded))

    elif len(batch[0]) == 3:
        #training
        batch.sort(key=lambda x: x[:][1].shape[0], reverse=True)
        keys = []
        lengths = []
        feats_padded = []
        label_padded = []
        max_len = max((batch[0][1].shape)[0], 99)

        for i in range(len(batch)):
            keys.append(batch[i][0])
            act_len = (batch[i][1].shape)[0]
            pad_len = max_len - act_len
            feats_padded.append(np.pad(batch[i][1], ((WINDOW_SIZE, pad_len), (0, 0)), "constant") )
            label_padded.append(batch[i][2])
            lengths.append(act_len)
        return keys, torch.from_numpy(np.array(lengths)), torch.from_numpy(np.array(feats_padded)), label_padded
        
    else:
        print("Error: we don't support this kind of datatype")
