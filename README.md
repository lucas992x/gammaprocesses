This is the code I use for my mastes thesis. If you find it useful for your research, please cite it!

## Quick info
A stochastic process is a gamma process if its increments are independent and gamma distributed with shape parameter `v(t)` (non-decreasing and right-continuous with `v(0) = 0`) and rate parameter `u > 0`. It can be used to approximate the degradation of objects.

`gammaprocesses.py` can import data from a dataset (two examples are given here) and compute the parameters, or generate random samples from arbitrary parameters and re-solve to compare results. It uses [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) and [Matplotlib](https://matplotlib.org/), that can be easily installed with [pip](https://pypi.org/project/pip/).

The script expects that the dataset contains inspection times and cumulative amounts of deterioration; if it contains increments of deterioration instead, the utility `incrtocumul.py` can be used to compute cumulative values and save them in a new text file.

## Acknowledgments
- Formulas that can compute the parameters from a dataset are described by J.M. van Noortwijk in "[A Survey of the application of Gamma processes in maintenance](https://www.researchgate.net/publication/222140978_A_Survey_of_the_application_of_Gamma_processes_in_maintenance)".
- Laser GaAs degradation dataset is described by W.Q. Meeker, L.A. Escobar in "[Statistical Methods for Reliability Data](https://www.researchgate.net/publication/261741677_Statistical_Methods_for_Reliability_Data_by_William_Q_Meeker_Luis_A_Escobar)".
- Fatigue-crack dataset is described by L.A. Rodriguez-Picón et al. in "[Degradation modeling based on gamma process models with random effects](https://www.researchgate.net/publication/316808032_Degradation_modeling_based_on_gamma_process_models_with_random_effects)".
