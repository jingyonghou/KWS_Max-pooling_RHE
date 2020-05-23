#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# Apache 2.0.

from __future__ import print_function

import os, sys, argparse, datetime, shutil
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from streaming_special_torch_dataset import * 
from kaldi_io import *
from RNNs import GRU
from TCN import TCN
from optimizer import get_std_opt
from utils import AverageMeter, count_parameters
from utils import str2bool

from mylosses import constrained_max_pooling_binary_OHEM_focal_ratio as loss_fn
from get_constraint import get_constraint_edge
from get_constraint import get_constraint_std
from spec_augment import spec_augment

constraint_fn_dict={"edge":get_constraint_edge,
        "std":get_constraint_std}

def get_args():
    """Get arguments from stdin."""
    parser = argparse.ArgumentParser(description='Pytorch acoustic model.')
    parser.add_argument('--encoder', type=str, default='gru',
                        help='encoder type {default: gru}')
    parser.add_argument('--random-n', type=str2bool, default=False,
            help='whether randomly select negative samples{default: False}')
    parser.add_argument('--spec-augment', type=int, default=0, metavar='n/y',
                        help='how many times doing spec_augmentation on the training sample (default: 0).')
    parser.add_argument('--gamma-p', type=float, default=0, metavar='n/y',
                        help='gamma of focal loss for positive samples (default: 0)')
    parser.add_argument('--gamma-n', type=float, default=0, metavar='n/y',
                        help='gamma of focal loss for negative samples (default: 0)')
    parser.add_argument('--clamp', type=float, default=0, metavar='n/y',
                        help='clamp after sigmoid to prevent NaN error (default: 0)')
    parser.add_argument('--constraint', type=int, default=0, metavar='n/y',
                        help='how many epoches adding constraints to the positive training sample (default: 0).')
    parser.add_argument('--constraint-type', type=str, default='edge', metavar='n/y',
                        help='constraint type(default: edge).')
    parser.add_argument('--cl', type=int, default=30, metavar='N',
                        help='left context of constraint(default: 30).')
    parser.add_argument('--cr', type=int, default=30, metavar='N',
                        help='right context of constraint(default: 30).')
    parser.add_argument('--ohem', type=int, default=10000, metavar='N',
                        help='ohem threshold (default: 10000).')
    parser.add_argument('--max-ratio', type=int, default=1, metavar='N',
                        help='max ratio of negative samples (default: 1).')
    parser.add_argument('--num-p', type=int, default=1, metavar='N',
                        help='Number of positive example used per utterance(default: 1).')
    parser.add_argument('--num-n', type=int, default=1, metavar='N',
                        help='Number of negative example used per utterance(default: 1).')
    parser.add_argument('--input-dim', type=int, default=40, metavar='N',
                        help='Input feature dimension without context (default: 40).')
    parser.add_argument('--kernel-size', type=int, default=3, metavar='N',
                        help='Kernel size of Wavenet or CNN (default:3).')
    parser.add_argument('--hidden-dim', type=int, default=128, metavar='N',
                        help='Hidden dimension of feature extractor (default: 128).')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
                        help='Numbers of hidden layers of feature extractor (default: 2).')
    parser.add_argument('--output-dim', type=int, default=1, metavar='N',
                        help='Output dimension, number of classes (default: 1).')
    parser.add_argument('--dropout', type=float, default=0.0001, metavar='DR',
                        help='dropout of feature extractor (default: 0.0001).')
    parser.add_argument('--left-context', type=int, default=5, metavar='N',
                        help='Left context length for splicing feature (default: 0).')
    parser.add_argument('--right-context', type=int, default=5, metavar='N',
                        help='Right context length for splicing feature (default: 0).')
    parser.add_argument('--max-epochs', type=int, default=20, metavar='N',
                        help='Maximum epochs to train (default: 20).')
    parser.add_argument('--min-epochs', type=int, default=0, metavar='N',
                        help='Minimum epochs to train (default: 0).')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='Batch size for training (default: 8).')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                        help='Initial learning rate (default: 0.001).')
    parser.add_argument('--optimizer', type=str, default="sgd", metavar='optimizer',
                        help='optimizer used for training.')
    parser.add_argument('--init-weight-decay', type=float, default=5e-5, metavar='WD',
                        help='Weight decay (L2 normalization) (default: 5e-5).')
    parser.add_argument('--halving-factor', type=float, default=0.5, metavar='HF',
                        help='Half factor for learning rate (default: 0.5).')
    parser.add_argument('--start-halving-impr', type=float, default=0.01, metavar='S',
                        help='Improvement threshold to half the learning rate (default: 0.01).')
    parser.add_argument('--end-halving-impr', type=float, default=0.001, metavar='E',
                        help='Improvement threshold to stop half learning rate (default: 0.001).')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='Random seed (default: 1234).')
    parser.add_argument('--use-cuda', type=int, default=1, metavar='C',
                        help='Use cuda (1) or cpu(0).')
    parser.add_argument('--multi-gpu', type=int, default=0, metavar='G',
                        help='Use multi gpu (1) or not (0).')
    parser.add_argument('--train', type=int, default=1,
                        help='Executing mode, train (1) or test (0).')
    parser.add_argument('--train-scp', type=str, default='',
                        help='Training data file.')
    parser.add_argument('--dev-scp', type=str, default='',
                        help='Development data file.')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Directory to output the model.')
    parser.add_argument('--load-model', type=str, default='',
                        help='Previous model to load.')
    parser.add_argument('--test', type=int, default=0,
                        help='Executing mode, 1 for test, 0 no test')
    parser.add_argument('--test-scp', type=str, default='',
                        help='Test data file.')
    parser.add_argument('--output-file', type=str, default='',
                        help='Test output file')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='How many batches to wait before logging training status.')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='How many workers used to load data')
    args = parser.parse_args()
    return args

def adjust_learning_rate(args, optimizer):
    """Half the learning rate when relative improvement is too low.
    Args:
        args: Arguments for training.
        optimizer: Optimizer for training.
    """
    args.learning_rate *= args.halving_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate

def one_epoch(epoch, args, model, device, data_loader, optimizer=None, is_train=False):
    """one epoch."""
    if is_train:
        tag="Train"
    else:
        tag="Val"

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_step = len(data_loader)
    for batch_idx, (utt_ids, act_lens, inputs, targets) in enumerate(data_loader):
        inputs, act_lens = inputs.to(device), act_lens.to(device)
        if args.spec_augment > 0 and is_train:
            for i in range(args.spec_augment):
                inputs = spec_augment(inputs, act_lens, specaug_t=30, specaug_f=20)
        # Forward pass
        batch_size = inputs.shape[0]
        outputs = model(inputs, act_lens)
        if args.constraint > epoch:
            # get constraint
            constraints = constraint_fn_dict[args.constraint_type](device, 
                                                    targets, args.cl, args.cr)
            loss, acc, num_training = loss_fn(outputs, 
                            act_lens, targets, gamma_n=args.gamma_n,
                            gamma_p=args.gamma_p, OHEM_Thr=args.ohem, 
                            max_ratio=args.max_ratio, random_n=args.random_n, 
                            constraints=constraints, clamp=args.clamp)
        else:
            loss, acc, num_training = loss_fn(outputs, 
                            act_lens, targets, gamma_n=args.gamma_n,
                            gamma_p=args.gamma_p, OHEM_Thr=args.ohem, 
                            max_ratio=args.max_ratio, random_n=args.random_n, 
                            constraints=None, clamp=args.clamp)
            
        if is_train:
        # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            #name, param=list(model.named_parameters())[1]
            #print('Epoch:[{}/{}], param name:{},\n param:'.format(epoch+1, args.max_epochs, name, param))
            optimizer.step()

        acc_meter.update(acc, len(utt_ids))
        loss_meter.update(loss.item(), num_training)

        if batch_idx % args.log_interval == 0:
            # there is bugs optimizer.get_lr()
            print('Epoch: [{}/{}], Step: [{}/{}], Lr: {:.6f} {} Loss: {:.6f}, {} Acc: {:.6f}% '
                  .format(epoch+1, args.max_epochs, batch_idx+1, total_step, 
                      optimizer.get_lr() if optimizer != None else 0.0, tag, loss_meter.cur, 
                          tag, acc_meter.cur))

    print('Epoch: [{}/{}], Average {} Loss: {:.6f}, Average {} Acc: {:.6f}%'
          .format(epoch+1, args.max_epochs, 
                  tag, loss_meter.avg, 
                  tag, acc_meter.avg))
    return float(loss_meter.avg)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    avg_loss = one_epoch(epoch, args, model, device, 
                        train_loader, optimizer, is_train=True)
    return avg_loss
    
def validate(args, model, device, dev_loader, epoch):
    """Cross validate the model."""
    model.eval()
    with torch.no_grad():
        avg_loss = one_epoch(epoch, args, model, device, 
                        dev_loader, optimizer=None, is_train=False)
    return avg_loss

def test(args, model, device, test_loader, output_file):
    write_post = open_or_fd(output_file, "wb")
    model.eval()
    with torch.no_grad():
        for batch_idx, (utt_ids, act_lens, data, target) in enumerate(test_loader):
            data = data.to(device)
            batch_size = data.shape[0]
            output = model(data, act_lens)
            output = torch.sigmoid(output).cpu().numpy()
            for i in range(len(utt_ids)):
                utt_id = utt_ids[i]
                end_idx = act_lens[i]
                sub_output = output[i, 0:end_idx, :]
                write_mat(write_post, sub_output, utt_id)
    print("Done, Time: {}".format(datetime.datetime.now()))
    write_post.close()

class Model(nn.Module):
    def __init__(self, encoder, cls):
        super(Model, self).__init__()
        self.encoder = encoder
        self.cls = cls

    def forward(self, data, lenghts):
        output = self.encoder(data, lenghts)
        output = self.cls(output)
        return output

def main():
    args = get_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.encoder == 'gru':
        encoder = GRU(input_size=args.input_dim, output_size=args.hidden_dim, 
            hidden_size=args.hidden_dim, num_layers=args.num_layers, 
            bias=True, batch_first=True, dropout=args.dropout, bidirectional=False, output_layer=True)
        classifier = nn.Linear(args.hidden_dim, args.output_dim)
    elif args.encoder == "tcn":
        encoder = TCN(layer_size=4,
                stack_size=args.num_layers,
                in_channels = args.input_dim,
                hid_channels = args.hidden_dim,
                kernel_size = 8,
                dropout=args.dropout)
        classifier = nn.Linear(args.hidden_dim, args.output_dim)

    model = Model(encoder,classifier).to(device)
    
    params = count_parameters(model)

    print("Num parameters: %d, Num Flops: %d\n"%(params,0))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                           momentum=0.9, weight_decay=args.init_weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.init_weight_decay)
    elif args.optimizer == "noam":
        # here the learning rate the peak learning rate
        optimizer = get_std_opt(model, 200, args.learning_rate)
    else:
        print("Error: we don't support this kind of optimizer\n")
        sys.exit(1)

    print("Training Arguments:\n {}".format(args))
    print("Training Model:\n {}".format(model))
    print("Training Optimizer:\n {}".format(optimizer))

    # Load previous trained model
    if args.load_model != '':
        print("=> Loading previous checkpoint to train: {}".format(args.load_model))
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        prev_val_loss = checkpoint['prev_val_loss']
    elif not args.train:
        sys.exit("Option --load-model should not be empty for testing.")
    else:
        print("=> No checkpoint found.")
        prev_val_loss = float('inf')

    # For training
    if args.train:
        if args.train_scp == '' or args.dev_scp == '':
            sys.exit("Options --train-scp and --dev-scp are required for training.")

        if args.save_dir == '':
            sys.exit("Option --save-dir is required to save model.")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        halving = 0
        best_model = args.load_model
        kwargs = {'num_workers': 3, 'pin_memory': True} if args.use_cuda else {}

        # Training data loader
        train_set = StreamingTorchDataset(args.train_scp, 
                ["kaldi_reader", "raw_list_reader"], 
                args.left_context,
                args.right_context)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)

        # Dev data loader
        dev_set = StreamingTorchDataset(args.dev_scp, 
                ["kaldi_reader", "raw_list_reader"], 
                args.left_context, 
                args.right_context)
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn)

        for epoch in range(args.max_epochs):
            cur_tr_loss = train(args, model, device, train_loader, optimizer, epoch)
            cur_val_loss = validate(args, model, device, dev_loader, epoch)
            rel_impr = (prev_val_loss - cur_val_loss) / prev_val_loss

            model_name = 'nnet_epoch' + str(epoch+1) + '_lr' \
                        + str(args.learning_rate) + '_tr' + str(cur_tr_loss) \
                        + '_cv' + str(cur_val_loss) + '.ckpt'
            model_path = args.save_dir + '/' + model_name

            if cur_val_loss < prev_val_loss:

                prev_val_loss = cur_val_loss
                torch.save({
                    'prev_val_loss': prev_val_loss,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_path)
                best_model = model_path

                print("Model {} accepted. Time: {}".format(model_name,
                                                           datetime.datetime.now()))

            else:
                print ("Model {} rejected. Time: {}".format(model_name,
                                                            datetime.datetime.now()))
                if best_model != '':
                    print("=> Loading best checkpoint: {}".format(best_model))
                    checkpoint = torch.load(best_model)
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    prev_val_loss = checkpoint['prev_val_loss']
                else:
                    sys.exit("Error training neural network.")

            # Stopping training criterion
            if halving and rel_impr < args.end_halving_impr:
                if epoch < args.min_epochs:
                    print("We were supposed to finish, but we continue as min_epochs"
                          .format(args.min_epochs))
                else:
                    print("Finished, too small relative improvement {}".format(rel_impr))
                    break

            # Start halving when improvement is low
            if rel_impr < args.start_halving_impr:
                halving = 1

            if halving:
                adjust_learning_rate(args, optimizer)
                print("Halving learning rate to {}".format(args.learning_rate))

        if best_model != args.load_model:
            final_model = args.save_dir + "/final.mdl"
            shutil.copyfile(best_model, final_model)
            print("Succeeded training the neural network: {}/final.mdl"
                  .format(args.save_dir))
        else:
            sys.exit("Error training neural network.")
    # For testing
    if args.test:
        # Test data loader
        if args.test_scp == '' or args.output_file == '':
            sys.exit("Options --test-scp and --output-file are required for testing")
        test_set = StreamingTorchDataset(args.test_scp,["kaldi_reader", "raw_list_reader"], args.left_context, args.right_context)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn)
        test(args, model, device, test_loader, args.output_file) 


if __name__ == '__main__':
    main()

