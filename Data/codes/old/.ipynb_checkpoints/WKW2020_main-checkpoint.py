    ## Main Program of WKW2020
## The order of operation is :
    # 1. given one varphi: calcualte optimal results for six regional scenarios (US, EU, OECD, World, China, OECD plus China) 
    # 2. a for-loop over different values of varphi
    # 3. iterate the above for all seven tax cases (baseline, unilateral, purete, puretc, puretp, EC_hybrid, EP_hybrid, PC_hybrid)
## The result is saved in one dataframe and then exported to "output/output_case().csv"


import numpy as np
import pandas as pd
from Func import minimization
from Func import callback
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.__version__
pd.__version__

## scenario switch
case=3;  # 2 means no trade in goods; 3 means trade in both energy and goods
logit = 0; # 1 means logit estimations of supply elasticity; 0 means fixed elasticities at 0.5

## import BAU values (seven regional scenarios in the order of US, EU, OECD, World, China, OECD plus China)
if case==2:
    df = pd.read_csv("../Data/output/BaselineCarbon_2015_noTradeinGoods.csv",index_col=['region_scenario','regionbase'],header='infer')
elif case==3:
    df = pd.read_csv("../Data/output/BaselineCarbon_2015.csv", index_col=['region_scenario','regionbase'],header='infer')
df['jxbar']=df['CeFH']/(df['CeFH'] + df['CeFF']);
df['jmbar']=df['CeHH']/(df['CeHH'] + df['CeHF']);

## choose which regional scenario to run (runs all if not executed)
    #df=df.drop([1,2,4,5,6,7])  

## parameter values
alpha = 0.85;           # labor share parameter in manufacturing
theta = 4;              # scopevec for comparative advantage
sigma = 1;      # elasticity of demand for each individual manufactured good j at Home
sigmastar = 1;  # elasticity of demand for each individual manufactured good j at Foreign
epsilonD = alpha + (1 - alpha) * sigma;  #Home's elasticity of demand for embodied energy
epsilonDstar = alpha + (1 - alpha) * sigmastar;  #Foreign's elasticity of demand for embodied energy
# beta = 2.274853;
# gamma= 0.784877595;
beta=1.892412;
gamma=0.807998928;
epsilonS = 0.5;  #Homes's energy supply elasticity: beta/(1 - beta)
epsilonSstar = 0.5;  #Foreign's energy supply elasticity: betastar/(1 - betastar)



tax_scenario= pd.DataFrame({'tax_sce': ['Baseline','Unilateral','purete','puretc','puretp','EC_hybrid','EP_hybrid','PC_hybrid','EPC_hybrid'], 'Base':[1,0,1,1,1,1,1,1,1]},index=[1, 2, 3, 4, 5, 6, 7, 8, 9])
# use for quick test: tax_scenario= pd.DataFrame({'tax_sce': 'EPC_hybrid', 'Base':1},index=[1])
def cases(tax_scenario, alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS,epsilonSstar, beta, gamma, logit):
    ParaList = (alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS,epsilonSstar, beta, gamma, logit)
    # varphilist = np.arange (1.6,2.5,0.1) # marginal damages
    varphilist = np.arange (0,20.1,0.2) # marginal damages

    # use for quick test: varphilist = [2] or varphilist = np.arange (1.7,2.5,0.1)
    varphilist = [2]
    output=[]
    for varphi in varphilist:    
        te = 0; #initial value of extraction tax for iteration
        tb_mat = [0, 1];  #initial value of  border adjustment and proportion of it (prop is mainly used for PC hybrid)

        ## calculate for optimal taxes by maximizing welfare
        tax_df=df.apply(minimization, axis=1, raw=False, args=(tb_mat, te, varphi, tax_scenario, ParaList))
        
        appended_df = pd.merge(df, tax_df, on=['region_scenario','regionbase'])

        ## back out other optimal values
        output_df = appended_df.apply(callback, axis=1, raw=False, result_type=None, args=(varphi, tax_scenario, ParaList))
        output_df = pd.merge(tax_df, output_df, on=['region_scenario','regionbase'])
        output.append(output_df)
        print(varphi)

    output = pd.concat(output, axis=0, join='outer',  ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False,copy=True)
    output.reset_index(level=0, inplace=True)
    output = output.sort_values(by=['region_scenario','varphi'])
    if tax_scenario['tax_sce']=='purete' or tax_scenario['tax_sce']=='EP_hybrid':
        output.te[output.Qe_prime==0]=output.pe+output.tb

    print(tax_scenario['tax_sce'])
    return output

## apply the above for each tax scenario
output_all=tax_scenario.apply(cases, axis=1, args=(alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS,epsilonSstar, beta, gamma, logit))

output_list=[]
for i in range(1,len(tax_scenario)+1):
    output_list.append(output_all.loc[i])
    
    
Outcomes = pd.concat(output_list, axis=0, join='outer', ignore_index=False, keys=tax_scenario['tax_sce'], levels=None, verify_integrity=False,copy=True)
Outcomes.reset_index(level=0, inplace=True)

if epsilonS == 0.5 and epsilonSstar == 0.5:
    Outcomes.to_csv('output/output_case{0}.csv'.format(case), header=True) 
elif epsilonS == 0.5 and epsilonSstar == 2:
    Outcomes.to_csv('output/output_case{0}_D_2.csv'.format(case), header=True) 
elif epsilonS == 2 and epsilonSstar == 0.5:
    Outcomes.to_csv('output/output_case{0}_2_D.csv'.format(case), header=True) 







