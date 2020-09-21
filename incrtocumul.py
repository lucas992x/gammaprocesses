import argparse, os.path, sys
from decimal import Decimal
# decimal is used because it usually avoids "mistakes" made with float
# for example, float can print something like 6.39999999998 instead of 6.4

# example: python incrtocumul.py --infile fatiguecrack_incr.txt --mode rows --sep ',' --outfile fatiguecrack.txt

parser = argparse.ArgumentParser()
parser.add_argument('--infile', default = 'data.txt')  # input file
parser.add_argument('--sep', default = ',')  # data separator
parser.add_argument('--mode', default = '')  # rows or columns, as in gammaprocesses.py
parser.add_argument('--outfile', default = 'data_new.txt')  # output file (will be overwritten)
args = parser.parse_args()

# read from file, like in gammaprocesses.py
if not os.path.isfile(args.infile):
    sys.exit('Error: file "{}" not found!'.format(args.infile))
with open(args.infile, 'r') as file:
    if args.mode == 'rows':
        lines = file.read().splitlines()
        t = lines[0].strip().split(args.sep)
        x = [[Decimal(xx) for xx in lines[j].strip().split(args.sep)] for j in range(1, len(lines))]
    elif args.mode == 'columns':
        t = []
        x = []
        with open(datafile) as file:
            for line in file:
                values = line.strip().split(args.sep)
                t.append(values[0])
                for j in range(1, len(values)):
                    if len(x) < j:
                        x.append([Decimal(values[j])])
                    else:
                        x[j - 1].append(Decimal(values[j]))
    else:
        sys.exit('Error: mode "{}" not recognized!'.format(args.mode))

# compute cumulative values and save them in a new text file
with open(args.outfile, 'w') as file:
    file.write(args.sep.join(t) + '\n')
    for xx in x:
        cumuls = [str(sum(xx[:j])) for j in range(1, len(xx) + 1)]
        file.write(args.sep.join(cumuls) + '\n')
