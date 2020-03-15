from __future__ import print_function
import os
import sys

import numpy as np
import argparse

from kaldi_io import *
from utils import str2bool

keyword_dict={"hixiaowen":0, 
            "nihaowenwen":1,
            "freetext":0,
            "1":0,
            "0":0
            }

keyword_list = ["hixiaowen", "nihaowenwen", "1"]

def smooth(mat, smooth_window=30):
    if (smooth_window <=1):
        return mat

    length, dim_size = mat.shape
    ans = []
    for idx in range(length):
        h_smooth = np.max([0, idx - smooth_window])
        sums  = np.sum(mat[h_smooth:idx+1,:], 0)
        #print(sums.shape)
        ans.append(sums/float(idx-h_smooth+1.0))
    return np.stack(ans)

def build_label_dict(label_scp_file):
    label_dict = {}
    for line in open(label_scp_file).readlines():
        wav_id, feature_scp, label = line.strip().split()
        label_dict[wav_id] = int(label.strip().split(",")[0])
    return label_dict

if __name__=="__main__":
    if len(sys.argv) < 4:
        print("USAGE:python %s output.ark keyword dev_scp score.txt"%sys.argv[0])
        exit(1)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('posterior_file', metavar='<posterior-file>', 
            type=str, help='posteriors from NN outputs')
    parser.add_argument('keyword', metavar='<keyword>', 
            type=str, help='get score for which keyword')
    parser.add_argument('label_scp', metavar='<label_scp>', 
            type=str, help='label scp file')
    parser.add_argument('score_file', metavar='<score_file>', 
            type=str, help='score file')

    parser.add_argument('--ignore-keyword', metavar = '<--ignore-keyword>',
            type = str2bool, help = 'whether ignore rest of keyword as negative data')
    parser.add_argument('--smooth-window', metavar = '<--smooth-window>',
            type = int, help = 'smooth window size')
    parser.add_argument('--shift', metavar = '<--shift>',
            type = int, default=0, help = 'shift')

    args = parser.parse_args()

    keyword = args.keyword
    label_dict = build_label_dict(args.label_scp)
    fid = open(args.score_file,"w")
    
    ignore_keyword = args.ignore_keyword

    for utt_id, mat in read_mat_ark(args.posterior_file):
        if label_dict[utt_id]-1 == int(keyword_dict[keyword]):
            fid.writelines("1")
        elif  (label_dict[utt_id]-1) >=0 and ignore_keyword:
            continue
        else:
            fid.writelines("0")
        fid.writelines(" %s"%utt_id.strip())
        mat = smooth(mat, args.smooth_window)
        for i in range(mat.shape[0]):
            fid.writelines(" %f"%mat[i, keyword_dict[keyword]+args.shift])
        fid.writelines("\n")

    fid.close()

