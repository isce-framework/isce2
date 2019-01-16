#!/usr/bin/env python3
import sys
import argparse
import os
def main():
    args = parse()
    wisdom0 = 'wisdom0'
    wisdom1 = 'wisdom1'
    which = 0
    for t in args.type:
        for p in args.place:
            for d in args.direction:
                size = args.sizes[0]
                while size <= args.sizes[1]:
                    if which == 0:
                        if args.action == 'new':
                            append = ''
                        elif args.action == 'append':
                            append = '-w ' +  args.file
                        else:
                            print('Error. Unrecognized action',args.action)
                            raise Exception
                    else:
                        append = '-w wisdom' + str(which%2)
                    command = 'fftwf-wisdom -n ' + append + ' -o wisdom' + str((which+1)%2) + ' ' + t + p + d + str(size)
                    print("command = ", command)
                    os.system(command)
                    #print(command)
                    size *= 2
                    which += 1
    os.system('mv wisdom' + str(which%2) + ' ' + args.file)
    os.system('rm wisdom' + str((which+1)%2))


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type = str, default = 'new', dest = 'action', help = 'What to do: new create a new wisdom file, appends it appends from the -f.')
    parser.add_argument('-f', '--file', type = str, default = 'isce_wisdom.txt', dest = 'file', help = 'File name for wisdom file.')
    parser.add_argument('-t', '--type', type = str, default = 'cr', dest = 'type', help = 'Type of fftw data c = complex r = real.')
    parser.add_argument('-p', '--place', type = str, default = 'io', dest = 'place', help = 'Type of fftw place i = in place o = out of place.')
    parser.add_argument('-d', '--direction', type = str, default = 'fb', dest = 'direction', help = 'Type of fftw direction f = forward b = backward.')
    parser.add_argument('-s', '--sizes', type = int,nargs = '+', default = [32,65536], dest = 'sizes', help = 'Min and max.')
    return parser.parse_args()

if __name__ == '__main__':
    sys.exit(main())
