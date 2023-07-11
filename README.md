

**Python files**:
1. simulation_results/codes/direct_energy_consumption/oucp_utils.py contains functions and their helpers to calculate optimal prices for various policies (7 constrained policies as well as unilateral optimal). Note that tax_scenario = "global" is under construction so there may be bugs.
In the same folder, oucp_master.ipynb is the master file that calls functions in oucp_utils.py. One can adjust parameters (sigma, theta, phi) or choose specific tax scenarios. However, to take advantage of ces production function, one needs to change alpha and rho variables in ces_renewable_util. To find alpha given rho, use the file calibrate_alpha.

2. simulation_results/codes/plotting_code/figures_paper.ipynb generates all figures for the paper. Note that output_case3.csv and ces0.csv are identical results, except output_case3 was ran earlier with more phi values.

3. raw_data contains carbon flow matrix for 3 scenarios: no trade in goods, trade in goods with no renewable, and trade in goods with 13.3% of renewable energy.

**Simulation results:**
All results have default parameters alpha = 0.02, rho = -1.22, sigma = sigma^* = 1.375, epsilonS = epsilonS^* = 0.5 unless otherwise specified.

4. output contains simulation results for various runs of the simulation.
     i. direct_consumption_constrained contains simulations for the constrained policies (combinations of extraction, production and consumption taxes) where Home country is defined as OECD.
   
     ii. direct_consumption_opt contains simulation results for the optimal unilateral policy, and various Home country scenarios (OECD, China, US, EU etc).

     iii. direct_consumption_renewable contains simulation results for a world with renewable energy (no carbon emissions).

     iv. direct_consumption_higheps_{opt, constrained} contains simulation where Foreign extraction elasticity is set to 2 as opposed to 0.5.
      

