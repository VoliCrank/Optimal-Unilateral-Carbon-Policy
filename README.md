*****Code File description*******

**Stata files:
1. BaselineCarbon_OECD_IEA.do uses raw data from OECD carbon embodied in goods and IEA energy extraction data to generate our baseline values of carbon matrix.

2. alpha.do calibrates the value of /alpha - output elasticity of labor.

3. epsilonS.do calibrates for /epsilon_s - energy supply elasticity.

**Python files:
1. data/codes/ces/ces_functional.py is the main program operating optimization and calculating optimal prices and taxes for seven tax scenarios (Baseline, unilateral*, pure consumption tax, pure extraction tax, pure production tax, Hybrid of consumption and extraction tax, and hybrid of consumption and production tax).

2. ces_fun_util.py is a file of utility functions. It contains all the functions necessary to compute optimal values.

3. Figures.py is the program to produce figures in the paper. It uses optimal results saved from the main optimization program.
