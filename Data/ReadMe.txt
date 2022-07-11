*****Code File description*******

**Stata files:
1. BaselineCarbon_OECD_IEA.do uses raw data from OECD carbon embodied in goods and IEA energy extraction data to generate our baseline values of carbon matrix.

2. alpha.do calibrates the value of /alpha - output elasticity of labor.

3. epsilonS.do calibrates for /epsilon_s - energy supply elasticity.

**Python files:
1. WKW2020_main.py is the main program operating optimization and calculating optimal prices and taxes for:
	a. five regional scenarios (US,EU,OECD,WORLD,CHINA as home country) 
	b. seven tax scenarios (Baseline, unilateral*, pure consumption tax, pure extraction tax, pure production tax, Hybrid of consumption and extraction tax, and hybrid of consumption and  production tax)

2. Func.py is a module being imported by WKW2020_main.py. It contains all the function forms.

3. Figures.py is the program to produce figures in the paper. It uses optimal results saved from the main optimization program (saved in output/output.csv and output/output_para.csv).


*****Command line execution*******
The shell script "run.sh" has been written for users who wants to replicate the study. To run the shell script:
1. open terminal
2. set directory to current folder:
'cd /YourPath'
3. Execute the shell script:
'Sh run.sh'




