import argparse, sys, os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm
from scipy.optimize import curve_fit, minimize, fsolve
from scipy.special import gamma, digamma

# Examples
# python gammaprocess.py --file gaaslasers_data.txt --sep ',' --mode rows --numsamples 15 --critical 10 | tee gaaslasers_results.txt
# python gammaprocess.py --file fatiguecrack_data.txt --sep ',' --mode rows --numsamples 10 --critical 0.4 | tee fatiguecrack_results.txt

# Notes and TODOs:
# Raises warnings like "RuntimeWarning: invalid value encountered in double_scalars" or "RuntimeWarning: divide by zero encountered in power"
# Has problems with b greater than 2 (correlated to the previous problem?)
# Should use better file names for graphs

# compute mean, variance and two percentiles of some data (default 2.5% and 97.5%)
class Stats:
    def __init__(self, data, percentile = 2.5):
        # remove some data to avoid a result "ruined" by a few mistakes
        if len(data) >= 10:
            data = sorted(data)
            remove = min([len(data) // 10, 5])
            self.values = data[remove:-remove]
        else:
            self.values = data
        self.n = len(self.values)
        self.mean = np.mean(self.values)
        sqsum = sum([(x - self.mean) ** 2 for x in self.values])
        self.variance = sqsum / self.n
        self.std = np.sqrt(sqsum / (self.n - 1))
        self.lowperc = np.percentile(self.values, percentile)
        self.upperc = np.percentile(self.values, 100 - percentile)

# read values from dataset, separated by 'sep' (comma, semicolon or whatever)
# return t = [t0, t1, ...] and x = [[x1], [x2], ...] where [xj] = [xj0, xj1, ...]
def ReadDataset(datafile, sep, mode):
    if not os.path.isfile(datafile):
        sys.exit('Error: file "{}" not found!'.format(datafile))
    # 'rows' mode: row 1 contains values of t, other rows values of x (one object per row)
    if mode == 'rows':
        with open(datafile) as file:
            lines = file.read().splitlines()
        t = [float(tt) for tt in lines[0].strip().split(sep)]
        x = [[float(xx) for xx in lines[j].strip().split(sep)] for j in range(1, len(lines))]
    # 'columns' mode: row j contains t[j], x1[j], x2[j], ..., xn[j]
    elif mode == 'columns':
        t = []
        x = []
        with open(datafile) as file:
            for line in file:
                values = line.strip().split(sep)
                t.append(float(values[0]))
                for j in range(1, len(values)):
                    if len(x) < j:
                        x.append([float(values[j])])
                    else:
                        x[j - 1].append(float(values[j]))
    else:
        sys.exit('Error: mode "{}" not recognized!'.format(mode))
    # add leading zeros if necessary
    if t[0] > 0:
        t = [0] + t
    xx = []
    for xj in x:
        if xj[0] > 0:
            xx.append([0] + xj)
        else:
            xx.append(xj)
    return t, xx

# loglikelihood function of the increments with respect to c and u
def loglike(params, *args):
    b, c, u = params
    t, delta = args
    w = [t[j] ** b - t[j - 1] ** b for j in range(1, len(t))]
    # to maximize f, minimize -f
    return -np.prod([(u ** (c * w[j])) * (delta[j] ** (c * w[j] - 1)) * np.exp(-1 * u * delta[j]) / (gamma(c * w[j])) for j in range(len(w))])

# maximum-likelihood estimator of c (used as constraint to solve using ML method)
def maxlikec(params, *args):
    b, c, u = params  # u unused here
    t, delta, xn = args
    w = [t[j] ** b - t[j - 1] ** b for j in range(1, len(t))]
    logarg = c * (t[-1] ** b) / xn
    # avoid errors if logarithm argument is not positive
    # it can happen, I guess it is during the search of minima?
    if logarg <= 0:
        # any non-zero value is ok to make the constraint unsatisfied
        return 666
    else:
        return sum([w[j] * (digamma(c * w[j]) - np.log(delta[j])) for j in range(len(w))]) - (t[-1] ** b) * np.log(logarg)

# maximum-likelihood estimator of u (used as constraint to solve using ML method)
def maxlikeu(params, *args):
    b, c, u = params
    tn, xn = args
    return u - c * (tn ** b) / xn

# solve using method of moments
def SolveMoments(t, x, b):
    n = len(t)
    delta = [x[j] - x[j - 1] for j in range(1, n)]
    deltabar = x[-1] / (t[-1] ** b)
    w = [t[j] ** b - t[j - 1] ** b for j in range(1, n)]
    u = x[-1] * (1 - sum([wj ** 2 for wj in w]) / (sum(w) ** 2)) / sum([(delta[j] - deltabar * w[j]) ** 2 for j in range(n - 1)])
    c = u * deltabar
    return c, u

# solve using method of maximum likelihood
# if values = [b0, c0, u0] all parameters are evaluated, with this values as initial guesses
# if values = [b0, c0] only c and u are evaluated, b0 is fixed and c0 initial guess
def SolveMaxLike(t, x, values):
    n = len(x)
    delta = [x[j] - x[j - 1] for j in range(1, n)]
    if len(values) == 3:
        mins = minimize(loglike, values, args = (t, delta), constraints = [
        {'type': 'eq', 'fun': maxlikec, 'args': (t, delta, x[-1])},
        {'type': 'eq', 'fun': maxlikeu, 'args': (t[-1], x[-1])} ],
        bounds = ((0, None), (0, None), (0, None)))
        if mins.success == True:
            return mins.x[0], mins.x[1], mins.x[2]  # b, c, u
        else:
            return 0, 0, 0
    else:
        b0, c0 = values
        w = [t[j] ** b0 - t[j - 1] ** b0 for j in range(1, n)]
        func = lambda c : sum([w[j] * (digamma(c * w[j]) - np.log(delta[j])) for j in range(n - 1)]) - (t[-1] ** b0) * np.log(c * (t[-1] ** b0) / x[-1])
        c = fsolve(func, c0)[0]
        u = c * (t[-1] ** b0) / x[-1]
        return c, u

# function that will be fitted to get an initial guess for exponent value
def expon(t, a, b):
    return a * (t ** b)

# solve using method of maximum likelihood
# first try to get an initial guess for b by fitting the previous function
# if this doesn't work, use the initial guesses passed as argument
# each value can lead to local minimimum, so they are compared to find the actual minimum
# 3rd sample of fatiguecrack breaks all: b0 = 1.064, c0 = 27.221, u0 = 92.883, b = 49.990, c = 765.704, u = 15.078
def TrySolveSingle(t, x, bguesses):
    params = curve_fit(expon, t, x)
    b0 = params[0][1]
    c0, u0 = SolveMoments(t, x, b0)  # get initial guesses for c and u that satisfy constrains
    b, c, u = SolveMaxLike(t, x, [b0, c0, u0])
    #print('b0 = {}, c0 = {}, u0 = {}\nb = {}, c = {}, u = {}'.format(b0, c0, u0, b, c, u))
    if b > 0 and c > 0 and u > 0:
        return b, c, u
    else:
        solutions = []
        for b0 in bguesses:
            c0, u0 = SolveMoments(t, x, b0)  # get initial guesses for c and u that satisfy constrains
            b, c, u = SolveMaxLike(t, x, [b0, c0, u0])
            if b > 0:
                solutions.append([b, c, u])
        if solutions == []:
            sys.exit('Error: no solutions found for {}!'.format(x))
        delta = [x[j] - x[j - 1] for j in range(1, len(x))]
        minimum = sys.float_info.max
        minsol = 3 * [minimum]
        for solution in solutions:
            llvalue = loglike(solution, t, delta)
            if llvalue < minimum:
                minimum = llvalue
                minsol = solution
        #print('b = {}, c = {}, u = {}\n----------'.format(*minsol))
        return minsol  # [b, c, u]

# try to solve with method of maximum likelihood
def TrySolve(t, xx, bguesses, percentile):
    b = []
    c = []
    u = []
    a = []
    for x in xx:
        sol = TrySolveSingle(t, x, bguesses)
        if sol[0] > 0:
            b.append(sol[0])
            c.append(sol[1])
            u.append(sol[2])
            a.append(sol[1] / sol[2])
    b = Stats(b, percentile = percentile)
    c = Stats(c, percentile = percentile)
    u = Stats(u, percentile = percentile)
    a = Stats(a, percentile = percentile)
    return b, c, u, a

# solve using both methods
# method of moments uses b obtained from method of maximum likelihood
def SolveBoth(t, xx, bguesses, percentiles):
    bML, cML, uML, aML = TrySolve(t, xx, bguesses, percentiles)
    cMom = []
    uMom = []
    aMom = []
    for x in xx:
        c, u = SolveMoments(t, x, bML.mean)
        cMom.append(c)
        uMom.append(u)
        aMom.append(c / u)
    cMom = Stats(cMom, percentile = percentiles)
    uMom = Stats(uMom, percentile = percentiles)
    aMom = Stats(aMom, percentile = percentiles)
    return bML, cML, uML, aML, cMom, uMom, aMom

# print results after computing c and u
def PrintResults2(method, c, u, a, percentile):
    print('{}: results'.format(method))
    print('                           c          u          a')
    print('Mean:               {:8.3f}   {:8.3f}   {:8.3f}'.format(c.mean, u.mean, a.mean))
    print('Variance:           {:8.3f}   {:8.3f}   {:8.3f}'.format(c.variance, u.variance, a.variance))
    print('{:4.1f}% percentile:   {:8.3f}   {:8.3f}   {:8.3f}'.format(percentile, c.lowperc, u.lowperc, a.lowperc))
    print('{:4.1f}% percentile:   {:8.3f}   {:8.3f}   {:8.3f}'.format(100 - percentile, c.upperc, u.upperc, a.upperc))
    print('')

# print results after computing b, c, u
def PrintResults3(method, b, c, u, a, percentile):
    print('{}: results'.format(method))
    print('                           b          c          u          a')
    print('Mean:               {:8.3f}   {:8.3f}   {:8.3f}   {:8.3f}'.format(b.mean, c.mean, u.mean, a.mean))
    print('Variance:           {:8.3f}   {:8.3f}   {:8.3f}   {:8.3f}'.format(b.variance, c.variance, u.variance, a.variance))
    print('{:4.1f}% percentile:   {:8.3f}   {:8.3f}   {:8.3f}   {:8.3f}'.format(percentile, b.lowperc, c.lowperc, u.lowperc, a.lowperc))
    print('{:4.1f}% percentile:   {:8.3f}   {:8.3f}   {:8.3f}   {:8.3f}'.format(100 - percentile, b.upperc, c.upperc, u.upperc, a.upperc))
    print('')

# generate random samples
def GenerateSamples(num, times, b, c, u):
    samples = []
    for j in range(num):
        sample = [0]
        for j in range(1, len(times)):
            incr = np.random.gamma(c * (times[j] ** b - times[j - 1] ** b), 1 / u)
            sample.append(incr + sample[-1])
        samples.append(sample)
    return samples

# plot graph
def PlotGraph(x, yy, title, xlimits = [], ylimits = [], critical = 0, labels = ['Time', 'Degradation']):
    if len(yy) == 2:
        # in this case plot normal pdf with a dotted line and estimated pdf with a continuous line
        plt.plot(x, yy[1], linestyle = ':')
        plt.plot(x, yy[0])
    else:
        # plot horizontal line with critical level
        if critical > 0:
            plt.axhline(critical)
        # plot all samples
        for y in yy:
            plt.plot(x, y)
    plt.grid()
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if xlimits:
        plt.xlim(*xlimits)
    if ylimits:
        plt.ylim(*ylimits)
    plt.title(title)
    #plt.show()
    plt.savefig('{}.png'.format(title.replace('\n', ' ')))
    plt.close()

# print a sample elegantly to satisfy my OCD
def PrintSample(sample, sep, pad, decs):
    print(sep.join(['{:{}.{}f}'.format(s, pad + 1 + decs, decs) for s in sample]).strip())

# print and/or plot samples
#Original samples
#Original samples, critical = 10.0
#Estimated pdf of failure time with 50 samples\nb = 1.400, c = 11.000, u = 6.000, a = 1.833, critical = 30.0
#Samples generated with arbitrary parameters\nb = 1.400, c = 11.000, u = 6.000, a = 1.833
#Samples generated with arbitrary parameters\nb = 1.400, c = 11.000, u = 6.000, a = 1.833, critical = 30.0
#Samples generated with parameters obtained from method of maximum likelihood\nb = 1.400, c = 11.000, u = 6.000, a = 1.833
#Samples generated with parameters obtained from method of maximum likelihood\nb = 1.400, c = 11.000, u = 6.000, a = 1.833, critical = 30.0
#Samples generated with parameters obtained from method of moments\nb = 1.400, c = 11.000, u = 6.000, a = 1.833, critical = 30.0
def PrintPlotSamples(t, samples, b, c, u, where, method = None, limits = [[], []], critical = 0):
    if method is None:
        title = 'Original samples'
    else:
        title = 'Samples generated with {}\nb = {:.3f}, c = {:.3f}, u = {:.3f}, a = {:.3f}'.format(method, b, c, u, c / u)
    if critical > 0:
        title += ', critical = {}'.format(critical)
    # print to console
    if where in ['console', 'both']:
        maxvalue = max(t[-1], max([v[-1] for v in samples]))
        decs = 3
        pad = len('{:.{}f}'.format(maxvalue, decs)) - 1 - decs
        print('\n{}:'.format(title))
        PrintSample(t, args.sep, pad, decs)
        for sample in samples:
            PrintSample(sample, args.sep, pad, decs)
    # plot to graph
    if where in ['graphs', 'both']:
        PlotGraph(t, samples, title, limits[0], limits[1], critical)

# given t, x and the critical value for x, estimate the time of failure
def GetFailureTime(t, x, critical):
    # failure has not happened yet
    if x[-1] < critical:
        failtime = None
    # time of failure is contained in t
    elif critical in x:
        failtime = t[x.index(critical)]
    # estimate time of failure
    else:
        index = x.index(max([xx for xx in x if xx < critical]))  # index of last inspection before failure
        prev = [t[index], x[index]]  # last inspection before failure
        next = [t[index + 1], x[index + 1]]  # first inspection after failure
        # apply a proportion to compute time of failure
        failtime = prev[0] + (next[0] - prev[0]) * (critical - prev[1]) / (next[1] - prev[1])
    return failtime

def GetFailurePdf(t, samples, critical, b = None, c = None, u = None, percentile = 2.5):
    # compute failure times
    failuretimes = []
    for sample in samples:
        failuretime = GetFailureTime(t, sample, critical)
        if failuretime is not None:
            failuretimes.append(failuretime)
    if failuretimes == []:
        print('No sample reached failure!')
    else:
        failuretimes = Stats(failuretimes, percentile = percentile)
        # plot the graph
        xgraph = np.linspace(0.9 * failuretimes.lowperc, 1.1 * failuretimes.upperc, num = 100)
        title = 'Estimated pdf of failure time with {} samples'.format(len(samples))
        # parameters will be printed if arbitrary
        if b is not None:
            title += '\nb = {:.3f}, c = {:.3f}, u = {:.3f}, a = {:.3f}'.format(b, c, u, c / u)
        title += ', critical = {}'.format(critical)
        # estimate pdf of failure time using Gaussian kernels
        kde = gaussian_kde(failuretimes.values)
        # normal distribution pdf, to compare it with estimated pdf
        normpdf = norm.pdf(xgraph, failuretimes.mean, failuretimes.std)
        # plot the two curves on the same graph
        PlotGraph(xgraph, [kde(xgraph), normpdf], title, [min(xgraph), max(xgraph)], [0, 1.05 * max([max(kde(xgraph)), max(normpdf)])], labels = ['', ''])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments needed to read a dataset and compute parameters
    parser.add_argument('--file', default = '', help = 'dataset file')
    parser.add_argument('--sep', default = ',', help = 'character that separates values in data file')
    parser.add_argument('--mode', default = 'rows', help = '"rows" or "columns"')
    parser.add_argument('--b0', default = 0, type = float, help = 'exponent if known')
    parser.add_argument('--percentiles', default = 2.5, type = float, help = 'percentiles to show for estimated parameters (default 2.5% and 97.5%, the second one is 100 minus the first one)')
    # arguments needed to generate random samples
    parser.add_argument('--numsamples', default = 0, type = int, help = 'number of samples to generate')
    parser.add_argument('--times', default = '', help = 'if generating samples from custom parameters instead of computed parameters, pass them and the times, separated by "sep" argument; a critical value can also be passed')
    parser.add_argument('--b', default = 0, type = float, help = '')
    parser.add_argument('--c', default = 0, type = float, help = '')
    parser.add_argument('--u', default = 0, type = float, help = '')
    parser.add_argument('--critical', default = 0, type = float, help = '')
    parser.add_argument('--plots', default = 'graphs', help = '"graphs" to plot graphs, "console" to print values, "both" to do both')
    # arguments needed to re-compute parameters after generating samples
    parser.add_argument('--resolve', default = 'no', help = '')
    args = parser.parse_args()
    # intializing stuff
    bguesses = [(j + 1) / 8 for j in range(48)]
    cMom = []
    uMom = []
    aMom = []
    bML = []
    cML = []
    uML = []
    aML = []
    samples = []
    # read data from dataset if argument is passed
    if args.file:
        t, xx = ReadDataset(args.file, args.sep, args.mode)
        print('{}: {} object(s) detected\n'.format(args.file, len(xx)))
        # if b0 is passed, both methods are used to compute parameters
        if args.b0:
            for x in xx:
                c1, u1 = SolveMoments(t, x, args.b0)
                c2, u2 = SolveMaxLike(t, x, [args.b0, c1 * np.random.uniform(0.95, 1.05)])
                cMom.append(c1)
                uMom.append(u1)
                aMom.append(c1 / u1)
                cML.append(c2)
                uML.append(u2)
                aML.append(c2 / u2)
            cMom = Stats(cMom, percentile = args.percentiles)
            uMom = Stats(uMom, percentile = args.percentiles)
            aMom = Stats(aMom, percentile = args.percentiles)
            cML = Stats(cML, percentile = args.percentiles)
            uML = Stats(uML, percentile = args.percentiles)
            aML = Stats(aML, percentile = args.percentiles)
            # print results
            PrintResults2('Method of moments', cMom, uMom, aMom, args.percentiles)
            PrintResults2('Method of maximum likelihood', cML, uML, aML, args.percentiles)
        # if b0 is not passed, b will be evaluated using ML method
        else:
            bML, cML, uML, aML, cMom, uMom, aMom = SolveBoth(t, xx, bguesses, args.percentiles)
            # print results
            PrintResults3('Method of maximum likelihood', bML, cML, uML, aML, args.percentiles)
            PrintResults2('Method of moments', cMom, uMom, aMom, args.percentiles)
    # generate random samples
    if args.numsamples:
        if args.plots not in ['graphs', 'console', 'both']:
            sys.exit('Error: "plot" argument not recognized!')
        # arbitrary inspection times
        if args.times:
            t = [float(tt) for tt in args.times.split(args.sep)]
            if t[0] > 0:
                t = [0] + t
            b = float(args.b)
            c = float(args.c)
            u = float(args.u)
            if b * c * u == 0 or b < 0 or c < 0 or u < 0:
                sys.exit('Error: insert all parameters with positive values!')
            print('Input values: b = {:.3f}, c = {:.3f}, u = {:.3f}, a = {:.3f}'.format(b, c, u, c / u))
            print('Input inspection times: ' + (args.sep + ' ').join([str(tt) for tt in t]))
            samples = GenerateSamples(args.numsamples, t, b, c, u)
            PrintPlotSamples(t, samples, b, c, u, args.plots, method = 'arbitrary parameters', limits = [[0, max(t)], [0, max([max(s) for s in samples])]], critical = args.critical)
        # inspection times read from dataset
        else:
            # plot graphs with same limits to have a better comparison
            if args.plots in ['graphs', 'both']:
                limits = [[0, max(t)], [0, max([max(x) for x in xx])]]
            else:
                limits = [[], []]
            if args.b0:
                b = args.b0
            else:
                b = bML.mean
            # print original samples
            PrintPlotSamples(t, xx, None, None, None, args.plots, method = None, limits = limits, critical = args.critical)
            # generate and print samples wih both methods
            samplesML = GenerateSamples(args.numsamples, t, b, cML.mean, uML.mean)
            PrintPlotSamples(t, samplesML, b, cML.mean, uML.mean, args.plots, method = 'parameters from method of maximum likelihood', limits = limits, critical = args.critical)
            samplesMom = GenerateSamples(args.numsamples, t, b, cMom.mean, uMom.mean)
            PrintPlotSamples(t, samplesMom, b, cMom.mean, uMom.mean, args.plots, method = 'parameters from method of moments', limits = limits, critical = args.critical)
    # if a critical value is passed, estimate pdf of failure time
    if args.critical:
        if args.file:
            GetFailurePdf(t, xx, args.critical, None, None, None, args.percentiles)
        else:
            GetFailurePdf(t, samples, args.critical, b, c, u, args.percentiles)
    # solve after generating the samples (used when they come from arbitrary parameters)
    if args.resolve == 'yes':
        bML, cML, uML, aML, cMom, uMom, aMom = SolveBoth(t, samples, bguesses, args.percentiles)
        # print results
        PrintResults3('Method of maximum likelihood', bML, cML, uML, aML, args.percentiles)
        PrintResults2('Method of moments', cMom, uMom, aMom, args.percentiles)
