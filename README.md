

**Python files**:
1. simulation_results/codes/simulation_code/ces_renewable_util.py contains functions and their helpers to calculate optimal prices for various policies (7 constrained policies as well as unilateral optimal). Note that tax_scenario = "global" is under construction so there may be bugs.
In the same folder, ces_renewable.ipynb is the master file that calls functions in ces_renewable_util.py. One can adjust parameters (sigma, theta, varphi) or choose specific tax scenarios. However, to take advantage of ces production function, one needs to change alpha and rho variables in ces_renewable_util. To find alpha given rho, use the file calibrate_alpha.

2. simulation_results/codes/plotting_code/figures_paper.ipynb generates all figures for the paper. Note that output_case3.csv and ces0.csv are identical results, except output_case3 was ran earlier with more phi values.

3. raw_data contains carbon flow matrix for 3 scenarios: no trade in goods, trade in goods with no renewable, and trade in goods with 13.3% of renewable energy.

**Simulation results:**
All results have default parameters alpha = 0.15, rho = 0, sigma = sigma^* = 1, epsilonS = epsilonS^* = 0.5 unless otherwise specified.

4. output contains simulation results for Cobb-Douglas production function (ces: rho = 0). output_case3 implies epsilonS = epsilonS^* = 0.5 while output_case3_D_2 are for epsilon = 0.5, epsilon^* = 2

5. output_sig0 contains simulation results for CES production function with various values of rho (rho = 0, 0.5 and -0.5).

6. output_renewable contains simulation result for assuming world starts with 13.3% renewable energy with the same level of CO2 emissions as before (scale all observations by 0.867).
