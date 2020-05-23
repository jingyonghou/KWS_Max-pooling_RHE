#!/usr/bin/python2

import argparse
import matplotlib
import os.path
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output_png', metavar = '<output-png>',
                        type = str, help = 'the output png file.')

    parser.add_argument('data', metavar = '<data>', nargs = '*',
                        type = str, help = 'data (label + ROC) to process.')

    # switches
    parser.add_argument('--x-min', metavar = 'x-min',
                        type = float, help='minimum value for x-axis.', default = 0.0)
    parser.add_argument('--x-max', metavar = 'x-max',
                        type = float, help='maximum value for x-axis.', default = 3.0)
    parser.add_argument('--y-min', metavar = 'y-min',
                        type = float, help='minimum value for y-axis.', default = 0.0)
    parser.add_argument('--y-max', metavar = 'y-max',
                        type = float, help='maximum value for y-axis.', default = 25.0)
    parser.add_argument('--show-roc', metavar = 'show-roc',
                        type = bool, help='to show ROC in GUI or not.', default = False)
    parser.add_argument('--line-style', metavar = 'line-style',
                        type = str, help='styles for each curve.', default = '')
    parser.add_argument('--color', metavar = 'color',
                        type = str, help='color for each curve.', default = '')

    args = parser.parse_args()

    if len(args.data) % 2 != 0:
        print('ERROR: invalid number of arguments, should be even (label + roc).')
        sys.exit(1)

    if not args.show_roc:  # disable GUI if desired.
        matplotlib.use('Agg')
    import matplotlib.pyplot as pyplot  # have to import after matplotlib.use().
    from pylab import subplot

    style = None
    if args.line_style:
        style = args.line_style.split()
    color = None
    if args.color:
        color = args.color.split()

    pyplot.figure(1)
    ax = subplot(111)
    ay = subplot(111)
    for i in xrange(len(args.data)/2):
        th = []
        fa = []
        fr = []
        roc_label = args.data[2 * i]
        roc_file = args.data[2 * i + 1]
        if not os.path.isfile(roc_file):
            print('WARNING: skipping label \"%s\", as file \"%s\" does not exist.'
                   % (roc_label, roc_file))
            continue
        for line in open(args.data[2 * i + 1], 'r'):
            token = line.rstrip().split()
            th.append(float(token[0]))
            fa.append(float(token[1]))
            fr.append(float(token[2]) * 100)
        pyplot.figure(1)
        if (color is not None) and (style is not None):
            pyplot.plot(fa, fr, style[i], color=color[i], label = roc_label)
        elif color is not None:
            pyplot.plot(fa, fr, color=color[i], label = roc_label)
        elif style is not None:
            pyplot.plot(fa, fr, style[i], label = roc_label)
        else:
            pyplot.plot(fa, fr, label = roc_label)
        pyplot.xlabel('False Alarm Per Hour', {'size':15})
        pyplot.ylabel('False Rejection Rate (%)', {'size':15})
        pyplot.xticks(fontsize=13)
        pyplot.yticks(fontsize=13)
        pyplot.xlim(args.x_min, args.x_max, )
        pyplot.ylim(args.y_min, args.y_max)
    pyplot.legend(loc = 'upper right', prop={'size':13})
    pyplot.grid(color='k',linestyle='-.')
    pyplot.savefig(args.output_png, dpi=800)
    if args.show_roc:
        pyplot.show()
