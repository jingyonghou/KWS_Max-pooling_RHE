#!/usr/bin/env python

# Copyright 2017 houjingyong@gmail.com 
#
# MIT licence
import sys

def build_scp_dict(scp_file):
    """
    Get a kaldi scp dict from a scp file
    """
    scp_dict = {}
    for line in open(scp_file).readlines():
        fields = line.strip().split()
        f_id = fields[0]
        f_value = " ".join(fields[1:])
        scp_dict[f_id] = f_value
    return scp_dict

def write_scp_dict(scp_file, scp_dict):
    fid = open(scp_file, "w")
    for utt_id in scp_dict.keys():
        fid.writelines(utt_id)
        fid.writelines(" ")
        fid.writelines(scp_dict[utt_id])
        fid.writelines("\n")
    fid.close()


if __name__=="__main__":
    if len(sys.argv) < 6:
        print("usage: python %s feats.scp labels.scp lens.scp max_lens feats_labels.scp\n"%(sys.argv[0]))
        exit(1)
    feats_scp_dict = build_scp_dict(sys.argv[1])
    labels_scp_dict = build_scp_dict(sys.argv[2])
    lens_scp_dict = build_scp_dict(sys.argv[3])
    max_lens = int(sys.argv[4])

    feats_labels_scp_dict={}
    feats_keys = feats_scp_dict.keys()
    labels_keys = labels_scp_dict.keys()

    num_remove=0    
    for utt_id in feats_keys:
        if labels_scp_dict.has_key(utt_id) and int(lens_scp_dict[utt_id]) < max_lens:
            feats_labels_scp_dict[utt_id] = feats_scp_dict[utt_id] + " " + labels_scp_dict[utt_id]
        elif int(lens_scp_dict[utt_id]) < max_lens:
            feats_labels_scp_dict[utt_id] = feats_scp_dict[utt_id] + " 0,-1,-1"
        else:
            num_remove+=1
    write_scp_dict(sys.argv[5],feats_labels_scp_dict)
    print("LOG: remove %d utterance for length or labels reason\n"%num_remove)
 
