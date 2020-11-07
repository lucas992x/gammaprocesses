import argparse, sys, os.path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm
from scipy.optimize import curve_fit, minimize, fsolve
from scipy.special import gamma, digamma

# Examples
# python3 gammaprocess.py --file fatiguecrack_data.txt --sep ',' --mode rows --numsamples 10 --critical 0.4 | tee results.txt
# python3 gammaprocess.py --file gaaslasers_data.txt --sep ',' --mode rows --numsamples 15 --critical 10 | tee results.txt
# python3 gammaprocess.py --numsamples 500 --times '1,2,3,4,5,6,7,8,9,10' --b 1.4 --c 11 --u 6 --plots graphs --critical 30 --resolve yes | tee results.txt
# python3 gammaprocess.py --numsamples 50 --times '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9' --b 1.02 --c 22.49 --u 59.81 --numsamples 50 --plots graphs --graphmax 0.495 --critical 0.4
# python3 gammaprocess.py --numsamples 50 --times '250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000' --b 0.998 --c 0.030 --u 14.25 --plots graphs --graphmax 12.21 --critical 10

# Notes and TODOs:
# Raises warnings like "RuntimeWarning: invalid value encountered in double_scalars" or "RuntimeWarning: divide by zero encountered in power"
# ML has problems with b greater than 2
# Try to fix c and u estimations by computing variance
# Change file names for graphs (maybe)
# Plot of estimated pdf is "shrinked" in some cases

# compute mean, variance, standard deviation and two percentiles of some data (default 2.5% and 97.5%)
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
        self.perc = percentile
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

# likelihood function of the increments with respect to c and u
def IncrLikelihood(params, *args):
    b, c, u = params
    t, delta = args
    w = [t[j] ** b - t[j - 1] ** b for j in range(1, len(t))]
    # to maximize f, minimize -f
    return -np.prod([(u ** (c * w[j])) * (delta[j] ** (c * w[j] - 1)) * np.exp(-1 * u * delta[j]) / (gamma(c * w[j])) for j in range(len(w))])

# maximum-likelihood estimator of c (used as constraint to solve using ML method)
def MaxLikeC(params, *args):
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
def MaxLikeU(params, *args):
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
# if guesses = [b0, c0, u0] all parameters are evaluated, with these values as initial guesses
# if guesses = [b0, c0] only c and u are evaluated, b0 is fixed and c0 initial guess
def SolveMaxLike(t, x, guesses):
    n = len(x)
    delta = [x[j] - x[j - 1] for j in range(1, n)]
    if len(guesses) == 3:
        mins = minimize(IncrLikelihood, guesses, args = (t, delta), constraints = [
        {'type': 'eq', 'fun': MaxLikeC, 'args': (t, delta, x[-1])},
        {'type': 'eq', 'fun': MaxLikeU, 'args': (t[-1], x[-1])} ],
        bounds = ((0, None), (0, None), (0, None)))
        if mins.success == True:
            return mins.x[0], mins.x[1], mins.x[2]  # b, c, u
        else:
            return 0, 0, 0
    else:
        b0, c0 = guesses
        w = [t[j] ** b0 - t[j - 1] ** b0 for j in range(1, n)]
        func = lambda c : sum([w[j] * (digamma(c * w[j]) - np.log(delta[j])) for j in range(n - 1)]) - (t[-1] ** b0) * np.log(c * (t[-1] ** b0) / x[-1])
        c = fsolve(func, c0)[0]
        u = c * (t[-1] ** b0) / x[-1]
        return c, u

# function that will be fitted
def Expon(t, a, b):
    return a * (t ** b)

# print results for given parameter
def PrintResults(param, desc):
    print(desc)
    print('            Mean     St. deviation  {:4.1f}% percentile  {:4.1f}% percentile'.format(param.perc, 100 - param.perc))
    print('{:16.3f}  {:16.3f}  {:16.3f}  {:16.3f}'.format(param.mean, param.std, param.lowperc, param.upperc))
    print('')

# solve to compute parameters b, c, u
def SolveAll3(t, xx, bguesses, percentiles, prnt = False):
    # initialize stuff
    bExp = []
    aExp = []
    bML = []
    cML = []
    uML = []
    cMLExp = []
    uMLExp = []
    cMomExp = []
    uMomExp = []
    cMomML = []
    uMomML = []
    for x in xx:
        # fit function
        params = curve_fit(Expon, t, x)
        a0, b0 = params[0]
        bExp.append(b0)
        aExp.append(a0)
        c0, u0 = SolveMoments(t, x, b0)  # also used as initial guesses that satisfy constrains
        cMomExp.append(c0)
        uMomExp.append(u0)
        # solve with method of maximum likelihood
        b1, c1, u1 = SolveMaxLike(t, x, [b0, c0, u0])
        if b1 > 0:
            solML = [b1, c1, u1]
        else:
            # try to solve with various guesses for b
            solutions = []
            for bg in bguesses:
                c0, u0 = SolveMoments(t, x, bg)  # get initial guesses for c and u that satisfy constrains
                b1, c1, u1 = SolveMaxLike(t, x, [bg, c0, u0])
                if b1 > 0:
                    solutions.append([b1, c1, u1])
            if solutions == []:
                solML = [0, 0, 0]
            else:
                # find the actual minimum among solutions (which can contain local minima)
                delta = [x[j] - x[j - 1] for j in range(1, len(x))]
                minimum = sys.float_info.max
                solML = 3 * [minimum]
                for solution in solutions:
                    llvalue = IncrLikelihood(solution, t, delta)
                    if llvalue < minimum:
                        minimum = llvalue
                        solML = solution
        bML.append(solML[0])
        cML.append(solML[1])
        uML.append(solML[2])
        c2, u2 = SolveMaxLike(t, x, [b0, c0])
        cMLExp.append(c2)
        uMLExp.append(u2)
        c3, u3 = SolveMoments(t, x, solML[0])
        cMomML.append(c3)
        uMomML.append(u3)
    # compute mean, variance and two percentiles for obtained values
    bExp = Stats(bExp, percentile = percentiles)
    aExp = Stats(aExp, percentile = percentiles)
    bML = Stats(bML, percentile = percentiles)
    cML = Stats(cML, percentile = percentiles)
    uML = Stats(uML, percentile = percentiles)
    cMLExp = Stats(cMLExp, percentile = percentiles)
    uMLExp = Stats(uMLExp, percentile = percentiles)
    cMomExp = Stats(cMomExp, percentile = percentiles)
    uMomExp = Stats(uMomExp, percentile = percentiles)
    cMomML = Stats(cMomML, percentile = percentiles)
    uMomML = Stats(uMomML, percentile = percentiles)
    # print all results
    if prnt == True:
        PrintResults(bExp, 'Parameter b computed fitting exponential function:')
        PrintResults(aExp, 'Parameter a computed fitting exponential function:')
        print('\n\n')
        PrintResults(bML, 'Parameter b computed with method of maximum likelihood:')
        PrintResults(cML, 'Parameter c computed with method of maximum likelihood:')
        PrintResults(uML, 'Parameter u computed with method of maximum likelihood:')
        print('a = c / u = {:.3f}'.format(cML.mean / uML.mean))
        print('\n\n')
        PrintResults(cMLExp, 'Parameter c computed with method of maximum likelihood and b fitted:')
        PrintResults(uMLExp, 'Parameter u computed with method of maximum likelihood and b fitted:')
        print('a = c / u = {:.3f}'.format(cMLExp.mean / uMLExp.mean))
        print('\n\n')
        PrintResults(cMomExp, 'Parameter c computed with method of moments and b fitted:')
        PrintResults(uMomExp, 'Parameter u computed with method of moments and b fitted:')
        print('a = c / u = {:.3f}'.format(cMomExp.mean / uMomExp.mean))
        print('\n\n')
        PrintResults(cMomML, 'Parameter c computed with method of moments and b from ML:')
        PrintResults(uMomML, 'Parameter u computed with method of moments and b from ML:')
        print('a = c / u = {:.3f}'.format(cMomML.mean / uMomML.mean))
    return bExp, aExp, bML, cML, uML, cMLExp, uMLExp, cMomExp, uMomExp, cMomML, uMomML

# solve to compute parameters c and u
def SolveAll2(t, xx, b, percentiles, prnt = False):
    # initialize stuff
    cMom = []
    uMom = []
    cML = []
    uML = []
    # compute parameters with both methods
    for x in xx:
        c1, u1 = SolveMoments(t, x, b)
        c2, u2 = SolveMaxLike(t, x, [b, c1 * np.random.uniform(0.95, 1.05)])
        cMom.append(c1)
        uMom.append(u1)
        cML.append(c2)
        uML.append(u2)
    # compute mean, variance and two percentiles for obtained values
    cMom = Stats(cMom, percentile = percentiles)
    uMom = Stats(uMom, percentile = percentiles)
    cML = Stats(cML, percentile = percentiles)
    uML = Stats(uML, percentile = percentiles)
    # print all results
    if prnt == True:
        PrintResults(cML, 'Parameter c computed with method of maximum likelihood:')
        PrintResults(uML, 'Parameter u computed with method of maximum likelihood:')
        print('a = c / u = {:.3f}'.format(cML.mean / uML.mean))
        print('\n\n')
        PrintResults(cMom, 'Parameter c computed with method of moments:')
        PrintResults(uMom, 'Parameter u computed with method of moments:')
        print('a = c / u = {:.3f}'.format(cMom.mean / uMom.mean))
    return cMom, uMom, cML, uML

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
    # adjust text size
    rcparams = {
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    }
    plt.rcParams.update(rcparams)
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
    elif len(failuretimes) == 1:
        print('Only 1 sample reached failure, at time {:.3f}'.format(failuretimes[0]))
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

def GenSamplesAndPlot3(numsamples, t, bML, cML, uML, bExp, cMLExp, uMLExp, cMomExp, uMomExp, cMomML, uMomML, limits, critical, plots):
    samplesML = GenerateSamples(numsamples, t, bML, cML, uML)
    PrintPlotSamples(t, samplesML, bML, cML, uML, plots, method = 'parameters\nfrom method of maximum likelihood', limits = limits, critical = critical)
    samplesMLExp = GenerateSamples(numsamples, t, bExp, cMLExp, uMLExp)
    PrintPlotSamples(t, samplesMLExp, bExp, cMLExp, uMLExp, plots, method = 'parameters\nfrom method of maximum likelihood (b fitted)', limits = limits, critical = critical)
    samplesMomExp = GenerateSamples(numsamples, t, bExp, cMomExp, uMomExp)
    PrintPlotSamples(t, samplesMomExp, bExp, cMomExp, uMomExp, plots, method = 'parameters\nfrom method of moments (b fitted)', limits = limits, critical = critical)
    samplesMomML = GenerateSamples(numsamples, t, bML, cMomML, uMomML)
    PrintPlotSamples(t, samplesMomML, bML, cMomML, uMomML, plots, method = 'parameters\nfrom method of method of moments (b from ML)', limits = limits, critical = critical)

def GenSamplesAndPlot2(numsamples, t, b, cML, uML, cMom, uMom, limits, critical, plots):
    samplesML = GenerateSamples(numsamples, t, b, cML, uML)
    PrintPlotSamples(t, samplesML, b, cML, uML, plots, method = 'parameters\nfrom method of maximum likelihood', limits = limits, critical = critical)
    samplesMom = GenerateSamples(numsamples, t, b, cMom, uMom)
    PrintPlotSamples(t, samplesMom, b, cMom, uMom, plots, method = 'parameters\nfrom method of moments', limits = limits, critical = critical)

# Akaike information criterion and Bayesian information criterion
def CalcCriterions(b, c, u, t, xx, bknown = False, method = ''):
    loglike = 0  # loglikelihood
    for x in xx:
        delta = [x[j] - x[j - 1] for j in range(1, len(x))]
        loglike += np.log(-IncrLikelihood([b, c, u], t, delta))
    if bknown == True:
        k = 2
    else:
        k = 3
    aic = 2 * (k - loglike)
    bic = k * np.log(len(xx)) - 2 * loglike
    if method:
        print('{}  {:8.3f}  {:8.3f}'.format(method, aic, bic))
    return aic, bic

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
    parser.add_argument('--graphmax', default = 0, type = float, help = 'upper limit for graphs (optional)')
    # arguments needed to re-compute parameters after generating samples
    parser.add_argument('--resolve', default = 'no', help = '"yes" to solve again after generating random samples')
    parser.add_argument('--bknown', default = 'no', help = '"yes" if b has to be treated as known')
    args = parser.parse_args()
    # intializing stuff
    bguesses = [(j + 1) / 8 for j in range(48)]
    samples = []
    # read data from dataset if argument is passed
    if args.file:
        t, xx = ReadDataset(args.file, args.sep, args.mode)
        print('{}: {} object(s) detected\n'.format(args.file, len(xx)))
        if args.b0:
            cMom, uMom, cML, uML = SolveAll2(t, xx, args.b0, args.percentiles, True)
        else:
            bExp, aExp, bML, cML, uML, cMLExp, uMLExp, cMomExp, uMomExp, cMomML, uMomML = SolveAll3(t, xx, bguesses, args.percentiles, True)
    # generate random samples
    if args.numsamples:
        if args.plots not in ['graphs', 'console', 'both']:
            sys.exit('Error: "plot" argument not recognized!')
        # arbitrary inspection times
        if args.times:
            t = [float(tt) for tt in args.times.split(args.sep)]
            if t[0] > 0:
                t = [0] + t
            if args.b * args.c * args.u == 0 or args.b < 0 or args.c < 0 or args.u < 0:
                sys.exit('Error: insert all parameters with positive values!')
            print('Input values: b = {:.3f}, c = {:.3f}, u = {:.3f}, a = {:.3f}'.format(args.b, args.c, args.u, args.c / args.u))
            print('Input inspection times: ' + (args.sep + ' ').join([str(tt) for tt in t]))
            print('')
            samples = GenerateSamples(args.numsamples, t, args.b, args.c, args.u)
            if args.graphmax:
                limits = [[0, max(t)], [0, args.graphmax]]
            else:
                limits = [[0, max(t)], [0, max([max(s) for s in samples])]]
            PrintPlotSamples(t, samples, args.b, args.c, args.u, args.plots, method = 'arbitrary parameters', limits = limits, critical = args.critical)
        # inspection times read from dataset
        else:
            # plot graphs with same limits to have a better comparison
            if args.plots in ['graphs', 'both']:
                if args.graphmax:
                    limits = [[0, max(t)], [0, args.graphmax]]
                else:
                    limits = [[0, max(t)], [0, max([max(x) for x in xx])]]
            else:
                limits = [[], []]
            # print original samples
            PrintPlotSamples(t, xx, None, None, None, args.plots, method = None, limits = limits, critical = args.critical)
            if args.b0:
                # generate and print samples with both methods
                GenSamplesAndPlot2(args.numsamples, t, args.b0, cML.mean, uML.mean, cMom.mean, uMom.mean, limits, args.critical, args.plots)
            else:
                GenSamplesAndPlot3(args.numsamples, t, bML.mean, cML.mean, uML.mean, bExp.mean, cMLExp.mean, uMLExp.mean, cMomExp.mean, uMomExp.mean, cMomML.mean, uMomML.mean, limits, args.critical, args.plots)
    # if a critical value is passed, estimate pdf of failure time
    if args.critical:
        if args.file:
            GetFailurePdf(t, xx, args.critical, None, None, None, args.percentiles)
        else:
            GetFailurePdf(t, samples, args.critical, args.b, args.c, args.u, args.percentiles)
    # solve after generating the samples (used when they come from arbitrary parameters)
    if args.resolve == 'yes':
        if args.bknown == 'yes':
            cMom, uMom, cML, uML = SolveAll2(t, samples, args.b, args.percentiles, True)
            GenSamplesAndPlot2(args.numsamples, t, args.b, cML.mean, uML.mean, cMom.mean, uMom.mean, limits, args.critical, args.plots)
        else:
            bExp, aExp, bML, cML, uML, cMLExp, uMLExp, cMomExp, uMomExp, cMomML, uMomML = SolveAll3 = SolveAll3(t, samples, bguesses, args.percentiles, True)
            GenSamplesAndPlot3(args.numsamples, t, bML.mean, cML.mean, uML.mean, bExp.mean, cMLExp.mean, uMLExp.mean, cMomExp.mean, uMomExp.mean, cMomML.mean, uMomML.mean, limits, args.critical, args.plots)
    # compute AIC, BIC, DIC
    if args.file or args.resolve == 'yes':
        print('\n\n\nAkaike information citerion, Bayesian information criterion and Deviance information criterion:')
        if args.b0:
            print(30 * ' ' + '       AIC       BIC       DIC')
            CalcCriterions(args.b0, cML.mean, uML.mean, t, xx, True, 'Method of maximum likelihood  ')
            CalcCriterions(args.b0, cMom.mean, uMom.mean, t, xx, True, 'Method of moments             ')
        elif args.bknown == 'yes':
            print(30 * ' ' + '       AIC       BIC       DIC')
            CalcCriterions(args.b, cML.mean, uML.mean, t, xx, True, 'Method of maximum likelihood  ')
            CalcCriterions(args.b, cMom.mean, uMom.mean, t, xx, True, 'Method of moments             ')
        else:
            print(40 * ' ' + '       AIC       BIC       DIC')
            CalcCriterions(bML.mean, cML.mean, uML.mean, t, xx, False, 'Method of maximum likelihood            ')
            CalcCriterions(bExp.mean, cMLExp.mean, uMLExp.mean, t, xx, False, 'Method of maximum likelihood (b fitted) ')
            CalcCriterions(bExp.mean, cMomExp.mean, uMomExp.mean, t, xx, False, 'Method of moments (b fitted)            ')
            CalcCriterions(bML.mean, cMomML.mean, uMomML.mean, t, xx, False, 'Method of moments (b from ML)           ')
