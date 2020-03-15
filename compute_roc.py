import os
import sys
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('score_file', metavar = '<score-file>',
                        type = str,  help = 'the file stored the score')
    parser.add_argument('det_file', metavar = '<det-file>',
                        type = str, help = 'det file used to draw det')
    parser.add_argument('time_neg', metavar = '<time-neg>',
                        type = float, help = 'negative sample duration')
    parser.add_argument('num_pos', metavar = '<num-pos>',
                        type = float, help = 'number of positive samples')

    parser.add_argument('--sliding-window', metavar = '<sliding-window-size>', 
                        type = int, help = 'sliding window, which descripted'
                        ' if there is a false alarm, how long we will scape')
    parser.add_argument('--start-threshold', metavar = 'start-threshold',
                        type = float, help = 'start threshold', default = 0.0)
    parser.add_argument('--end-threshold', metavar = 'end-threshold',
                        type = float, help = 'end threshold', default = 1.0)
    parser.add_argument('--threshold-step', metavar = 'step',
                        type = float, help = 'steps of threshold', 
                        default = 0.001)

    args = parser.parse_args()

    fscore = args.score_file
    froc = args.det_file
    fid = open(froc,"w")
    time_neg = args.time_neg # hour of negative sample
    num_pos  = args.num_pos # number of positive sample

    sliding_window = args.sliding_window
    threshold_step = args.threshold_step

    threshold = args.start_threshold
    end_threshold = args.end_threshold

    lines = []
    false_alarm_pre = -1
    lines = open(fscore).readlines()
    while threshold <= end_threshold + threshold_step:
        false_alarm  = 0.0
        false_reject = 0.0
        if threshold > 0.8:
            threshold_step = 0.003
        for line in lines:
            label = int(line.strip().split()[0])
            dec_score = [ float(x) for x in line.strip().split()[2:]]
            
            if label == 1:
                if max(dec_score) < threshold:
                    false_reject += 1
            else:
                tag = 0
                for x in dec_score:
                    if tag > 0: 
                        tag -= 1
                        continue
                    if float(x) >= threshold: 
                        #print(line_dec[0], float(i))
                        false_alarm += 1
                        tag = sliding_window
        fid.write('%f %f %f\n' % (threshold, false_alarm/time_neg, false_reject/num_pos))
        print(threshold, false_alarm/time_neg, false_reject/num_pos);sys.stdout.flush()
        false_alarm_pre = false_alarm
        threshold += threshold_step
    fid.close()
