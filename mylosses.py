import torch
import torch.nn.functional as F

import random
import numpy as np

from ohem import OHEM
def select_training_samples(new_outputs, new_targets, ratio=1, gamma_n=0.0, 
                            gamma_p=0.0, random_n=False):
    num_training=0
    num_positive=0
    num_negative=0
    loss = 0
    for i in range(len(new_outputs)):
        output = torch.cat(new_outputs[i])
        target = torch.LongTensor(np.concatenate(new_targets[i]))
        # how many positive targets
        positive_index = (target >= 1) # the label of positive label is 1
        negative_index = (target < 1) # the label of negative label is 0
        num_p = torch.sum(positive_index)
        selected_p_output = output[positive_index]
        weight = torch.pow(1.0-selected_p_output, gamma_p)
        loss += torch.sum(-weight * torch.log(selected_p_output))

        all_n_output = output[negative_index]
        num_n = min(int(ratio*num_p),len(all_n_output))
        if random_n:
            a = range(len(all_n_output))
            np.random.shuffle(a)
            sorted_index = torch.LongTensor(a)
        else:
            sorted_output, sorted_index = torch.sort(all_n_output, descending=True)
        selected_n_output = all_n_output[sorted_index[:num_n]]
        weight = torch.pow(selected_n_output, gamma_n)
        loss += torch.sum(-weight * torch.log(1.0-selected_n_output))
        num_training += (len(selected_p_output) + len(selected_n_output))
        num_positive += len(selected_p_output)
        num_negative += len(selected_n_output)
#    print(float(num_negative)/float(num_positive))
    return loss/num_training, num_training

def constrained_max_pooling_binary_OHEM_focal_ratio(all_output, num_samples, 
                                        all_target, gamma_n=0, gamma_p=0,
                                        OHEM_Thr=10000, max_ratio=1, random_n=False, 
                                        constraints=None, clamp=0):
    num_hit = 0
    # Here we clamp the sigmoid output to prevent NaN problem 
    # When we calculate loss
    all_output = torch.clamp(torch.sigmoid(all_output), clamp, 1.0-clamp)
    num_utts = all_output.shape[0]
    num_sigmoid = all_output.shape[2]

    new_outputs = []
    new_targets = []
    for j in range(num_sigmoid):
        new_outputs.append([])
        new_targets.append([])

    for i in range(num_utts):
        end_idx = num_samples[i]
        
        sorted_output, sorted_index = torch.sort(all_output[i, :end_idx], dim=0)
        reversed_index = torch.flip(sorted_index, dims=[0])
        if all_target[i][0] == 0:
        # target is 0, so we don't need constraint
            for j in range(num_sigmoid):
                selected_indexes = OHEM(reversed_index[:,j], OHEM_Thr)
                new_outputs[j].append(all_output[i, selected_indexes, j])
                new_targets[j].append([0]*len(selected_indexes))

            if torch.sum(sorted_output[-1,:] >= 0.5) <= 0: 
                # all the binary probilities are smaller than 0.5
                num_hit += 1
        else:
        # target is not 0, we should calculate constraint, here we support multiple constraint
            index_constraint = set()
            if constraints != None:
                for x in constraints[i]:
                    index_constraint = set.union(index_constraint, set(range(x[0], x[1])))
            # we calculate negative loss for all non-target sigmoid
            target_sigmoid = all_target[i][0]-1
            non_target_sigmoids = range(num_sigmoid)
            non_target_sigmoids.remove(target_sigmoid)
            for j in non_target_sigmoids:
                selected_indexes = OHEM(reversed_index[:,j], OHEM_Thr)
                new_outputs[j].append(all_output[i, selected_indexes, j])
                new_targets[j].append([0]*len(selected_indexes))
            # calculate positive loss for target sigmoid
            if len(index_constraint)==0:
            # non-constraint. constraints == None or this utterance don't have
            # constraint (is possible)
                new_outputs[target_sigmoid].append(sorted_output[-1, [target_sigmoid]])
                new_targets[target_sigmoid].append([1])
                if sorted_output[-1, target_sigmoid] >= 0.5:
                    num_hit += 1
            else:
            # with constraint. constraints != None and this utterance do have 
            # constraint
                index_constraint = torch.tensor(list(index_constraint))
                sorted_output_short, _ = torch.sort(all_output[i, index_constraint, target_sigmoid])
                new_outputs[target_sigmoid].append(sorted_output_short[-1].view(1,))
                new_targets[target_sigmoid].append([1])
                if sorted_output_short[-1] >= 0.5:
                    num_hit += 1
        if torch.sum(torch.isnan(sorted_output)) > 0:
            print("Error: output NaNs\n")
            exit(1)

    # Here we select training samples acorrding to max_ratio
    loss, num_training= select_training_samples(new_outputs, new_targets, ratio=max_ratio,
                                    gamma_n=gamma_n, gamma_p=gamma_p, 
                                    random_n=random_n)
    if torch.isnan(loss) > 0:
        print("Error: Loss NaNs\n")
        exit(1)
    return loss, float(num_hit)*100/len(num_samples), num_training

