*****Code File description*******

**Python files:
1. simulation_results/codes/simulation_code/ces_renewable_util.py contains functions and their helpers to calculate optimal prices for various policies (7 constrained policies as well as unilateral optimal). Note that tax_scenario = "global" is under construction so there may be bugs.

In the same folder, ces_renewable.ipynb is the master file that calls functions in ces_renewable_util.py. One can adjust parameters (sigma, theta, varphi) or choose specific tax scenarios. However, to take advantage of ces production function, one needs to change alpha and rho variables in ces_renewable_util. To find alpha given rho, use the file calibrate_alpha.

2. simulation_results/codes/plotting_code/figures_paper.ipynb generates all figures for the paper. Note that output_case3.csv and ces0.csv are identical results, except output_case3 was ran earlier with more phi values.

