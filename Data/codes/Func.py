import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.integrate import quad
#  The function "equilibrium_hatalgebra" contained equilibrium conditions, which allows us to calculate the equilibrium price with given set of taxes
#  The function "minuswelfare" contained equations to calculate Home's welfare.
    ## The only difference between the two minuswelfare function is the fisrt argument input: I do this because scipy.minimize cosiders the first argument as the unkown to be solved by default.
#  The function "minimization" is used to maximize Home's welfare to get optimal tax values and energy price.
#  The function "callback_pe" takes the tax values calculated by welfare maximization to back out corresponding energy price.
#  The function "callback" takes optimal taxes and energy price to recalculate all other values including world emissions, home emissions...

def equilibrium_hatalgebra(pe,*data):
    tb_mat, te, varphi, tax_scenario, ParaList, df = data
    alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS,epsilonSstar, beta, gamma, logit = ParaList
    
    ## optimal values
    # jxbar_hat =   (1 - df['jxbar']) ** (-1) / (((1 - df['jxbar']) ** (-1) - 1) + (1 + (1 - alpha) * tb_mat[0]/pe) ** (-theta) * (1 + tb_mat[0]/pe) ** ((1 - alpha) * theta));
    jxbar_hat = pe**(-alpha*theta) * (pe+tb_mat[0])**(-(1-alpha)*theta) / ( df['jxbar'] * pe**(-alpha*theta) * (pe+tb_mat[0])**(-(1-alpha)*theta) + (1-df['jxbar']) * (pe + (1-alpha) * tb_mat[0])**(-theta));
    j0_hat = (pe+tb_mat[0])**(-(1-alpha)*theta) / (df['jxbar'] * (pe+tb_mat[0])**(-(1-alpha)*theta) + (1-df['jxbar']) * pe**(-(1-alpha)*theta) );
    jmbar_hat = 1 ;
    
    if tax_scenario['tax_sce']=='Unilateral':
        te=varphi;
        tb_mat[1]=1;
        
    if tax_scenario['tax_sce']=='purete':
        jxbar_hat = 1;   
        jmbar_hat = 1;

    
    if tax_scenario['tax_sce']=='puretc':
        te=tb_mat[0];
        jxbar_hat = 1;
        jmbar_hat = 1;
        tb_mat[1]=1;
    
    if tax_scenario['tax_sce']=='puretp':
        te=tb_mat[0];
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        tb_mat[1]=1;

    
    if tax_scenario['tax_sce']=='EC_hybrid':
        te=varphi;
        jxbar_hat = 1;
        jmbar_hat = 1;
        tb_mat[1]=1;

    
    if tax_scenario['tax_sce']=='PC_hybrid':
        te=tb_mat[0];
        if tb_mat[1]<0:
            tb_mat[1]=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[2] * tb_mat[0])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0])/(pe + tb_mat[1] * tb_mat[0])) ** (theta * (1 - alpha)));
    
    if tax_scenario['tax_sce']=='EP_hybrid':
        te=tb_mat[1];
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));

    if tax_scenario['tax_sce']=='EPC_hybrid':
        te=varphi;
        if tb_mat[1]<0:
            tb_mat[1]=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[2] * tb_mat[0])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0])/(pe + tb_mat[1] * tb_mat[0])) ** (theta * (1 - alpha)));
               
    
    jxbar_prime = jxbar_hat * df['jxbar'];
    jmbar_prime = jmbar_hat * df['jmbar'];
    j0_prime = j0_hat * df['jxbar'];

    def tempFunction(i, theta, sigmastar):
        return (i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1)) 
    
    Bfunvec1_prime = quad(tempFunction,0,j0_prime, args=(theta, sigmastar))[0];
    Bfunvec2_prime = quad(tempFunction,0,jxbar_prime, args=(theta, sigmastar))[0];

    #if te is too large, HoGe stop producing
    petbte = pe + tb_mat[0] - te;
    z = pe + tb_mat[0] >= te;
    petbte = petbte * z;

    Qe_hat = (petbte) ** epsilonS;
    Qestar_hat = pe ** epsilonSstar;
          
    if logit==1:
        epsilonS=beta*(1-gamma)/(1-gamma+gamma*petbte**beta);
        epsilonSstar=beta*(1-gamma)/(1-gamma+gamma*pe**beta);
        Qe_hat = (petbte)**beta/(1-gamma+gamma*(petbte)**beta);
        Qestar_hat = pe**beta/(1-gamma+gamma*pe**beta);
 
    Qe_prime = df['Qe'] * Qe_hat;
    Qestar_prime = df['Qestar'] * Qestar_hat;
    
    
    CeHH_hat = (pe + tb_mat[0]) ** (-epsilonD) * jmbar_hat ** (1 + (1 - sigma)/theta);
    CeHH_prime = df['CeHH'] * CeHH_hat;
       
    
    # CeFH_hat = (1 + (1 - sigmastar)/theta) * pe ** (-(1 - alpha) * sigmastar) * (pe + tb_mat[0]) ** (-alpha) * Bfunvec_prime/(df['jxbar'] ** (1 +1/theta)) * (1 - df['jxbar']) ** (sigmastar/theta);
    CeFH1_hat = (pe +tb_mat[0])**(-epsilonDstar) * j0_hat**(1 + (1 - sigmastar)/theta);
    CeFH2_hat = (1 + (1 - sigmastar)/theta) * ((1-df['jxbar'])/df['jxbar'])**(sigmastar/theta) * pe**(-epsilonDstar) * (1 + tb_mat[0]/pe)**(-alpha) * (Bfunvec2_prime - Bfunvec1_prime)/df['jxbar']**(1+(1-sigmastar)/theta);
    CeFH1_prime = df['CeFH'] * CeFH1_hat;
    CeFH2_prime = df['CeFH'] * CeFH2_hat;
    CeFH_hat = CeFH1_hat + CeFH2_hat;
    if tax_scenario['Base']==1:
        CeFH_hat = pe ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeFH_hat = (pe + tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeFH_hat = (pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if np.isnan(CeFH_hat)==True:
        CeFH_hat=0
    CeFH_prime =df['CeFH'] * CeFH_hat;
    
    
    CeHF_hat = (pe + tb_mat[0]) ** (-epsilonD);
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeHF_hat = (pe) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeHF_hat = (pe + tb_mat[1] * tb_mat[0]) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    CeHF_prime = df['CeHF'] * CeHF_hat;
    
    
    CeFF_prime = df['CeFF'] * ((1 - jxbar_prime)/(1-df['jxbar'])) ** (1 + (1 - sigmastar)/theta) * pe ** (-epsilonDstar);
    
    diff = Qe_prime + Qestar_prime - (CeHH_prime + CeFH_prime + CeHF_prime + CeFF_prime);
    #print('diff='+str(diff))
    return diff
    

def minuswelfare(tb_mat, te, varphi, tax_scenario, ParaList, df):
     #solve for equilibrium
    data = (tb_mat, te, varphi, tax_scenario, ParaList, df)
    pe = fsolve(equilibrium_hatalgebra,1,args=data, full_output = True)
    if (pe[2] != 1):
        print("pe is", pe)
        print("tb is", tb_mat, "te is", te)
        print("tax is", tax_scenario['tax_sce'])
        print("opt value is", equilibrium_hatalgebra(pe[0][0], *data))
    pe=pe[0][0]
    # print('pe='+str(pe))
    if pe-te<0 and tax_scenario['tax_sce']=='purete':
        pe=te;
        
    # if pe+tb_mat[0]-tb_mat[1]<=0 and tax_scenario['tax_sce']=='EP_hybrid':
    #     # print(tb_mat)
    #     # print(pe)
    #     # tb_mat[0]=tb_mat[1]-pe;
    #     tb_mat[1]=pe+tb_mat[0];
    #     te=tb_mat[1];
 
    alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    # print(tb_mat)
     ## optimal values
    # jxbar_hat =   (1 - df['jxbar']) ** (-1) / (((1 - df['jxbar']) ** (-1) - 1) + (1 + (1 - alpha) * tb_mat[0]/pe) ** (-theta) * (1 + tb_mat[0]/pe) ** ((1 - alpha) * theta));
    jxbar_hat = pe**(-alpha*theta) * (pe+tb_mat[0])**(-(1-alpha)*theta) / ( df['jxbar'] * pe**(-alpha*theta) * (pe+tb_mat[0])**(-(1-alpha)*theta) + (1-df['jxbar']) * (pe + (1-alpha) * tb_mat[0])**(-theta));
    j0_hat = (pe+tb_mat[0])**(-(1-alpha)*theta) / (df['jxbar'] * (pe+tb_mat[0])**(-(1-alpha)*theta) + (1-df['jxbar']) * pe**(-(1-alpha)*theta) );
    jmbar_hat = 1 ;
    
    if tax_scenario['tax_sce']=='Unilateral':
        te=varphi;
        tb_mat[1]=1;
        
    if tax_scenario['tax_sce']=='purete':
        jxbar_hat = 1;   
        jmbar_hat = 1;

    
    if tax_scenario['tax_sce']=='puretc':
        te=tb_mat[0];
        jxbar_hat = 1;
        jmbar_hat = 1;
        tb_mat[1]=1;

    if tax_scenario['tax_sce']=='puretp':
        te=tb_mat[0];
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        tb_mat[1]=1;

    
    if tax_scenario['tax_sce']=='EC_hybrid':
        te=varphi;
        jxbar_hat = 1;
        jmbar_hat = 1;
        tb_mat[1]=1;

    
    if tax_scenario['tax_sce']=='PC_hybrid':
        te=tb_mat[0];
        if tb_mat[1]<0:
            tb_mat[1]=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[2] * tb_mat[0])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0])/(pe + tb_mat[1] * tb_mat[0])) ** (theta * (1 - alpha)));
    
    
    if tax_scenario['tax_sce']=='EP_hybrid':
        te=tb_mat[1];
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
   
    if tax_scenario['tax_sce']=='EPC_hybrid':
        te=varphi;
        if tb_mat[1]<0:
            tb_mat[1]=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[2] * tb_mat[0])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0])/(pe + tb_mat[1] * tb_mat[0])) ** (theta * (1 - alpha)));
    
                
    
    jxbar_prime = jxbar_hat * df['jxbar'];
    jmbar_prime = jmbar_hat * df['jmbar'];
    j0_prime = j0_hat * df['jxbar'];

    def tempFunction(i, theta, sigmastar):
        return (i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1)) 
    
    Bfunvec1_prime = quad(tempFunction,0,j0_prime, args=(theta, sigmastar))[0];
    Bfunvec2_prime = quad(tempFunction,0,jxbar_prime, args=(theta, sigmastar))[0];

    #if te is too large, Home stop producing
    petbte = pe + tb_mat[0] - te;
    z = pe + tb_mat[0] >= te;
    petbte = petbte * z;

    Qe_hat = (petbte) ** epsilonS;
    Qestar_hat = pe ** epsilonSstar;
          
    if logit==1:
        epsilonS=beta*(1-gamma)/(1-gamma+gamma*petbte**beta);
        epsilonSstar=beta*(1-gamma)/(1-gamma+gamma*pe**beta);
        Qe_hat = (petbte)**beta/(1-gamma+gamma*(petbte)**beta);
        Qestar_hat = pe**beta/(1-gamma+gamma*pe**beta);
 
    Qe_prime = df['Qe'] * Qe_hat;
    Qestar_prime = df['Qestar'] * Qestar_hat;
     
    CeHH_hat = (pe + tb_mat[0]) ** (-epsilonD) * jmbar_hat ** (1 + (1 - sigma)/theta);
    CeHH_prime = df['CeHH'] * CeHH_hat;
       
    
    # CeFH_hat = (1 + (1 - sigmastar)/theta) * pe ** (-(1 - alpha) * sigmastar) * (pe + tb_mat[0]) ** (-alpha) * Bfunvec_prime/(df['jxbar'] ** (1 +1/theta)) * (1 - df['jxbar']) ** (sigmastar/theta);
    CeFH1_hat = (pe +tb_mat[0])**(-epsilonDstar) * j0_hat**(1 + (1 - sigmastar)/theta);
    CeFH2_hat = (1 + (1 - sigmastar)/theta) * ((1-df['jxbar'])/df['jxbar'])**(sigmastar/theta) * pe**(-epsilonDstar) * (1 + tb_mat[0]/pe)**(-alpha) * (Bfunvec2_prime - Bfunvec1_prime)/df['jxbar']**(1+(1-sigmastar)/theta);
    CeFH1_prime = df['CeFH'] * CeFH1_hat;
    CeFH2_prime = df['CeFH'] * CeFH2_hat;
    CeFH_hat = CeFH1_hat + CeFH2_hat;
    if tax_scenario['Base']==1:
        CeFH_hat = pe ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeFH_hat = (pe + tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeFH_hat = (pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if np.isnan(CeFH_hat)==True:
        CeFH_hat=0
    CeFH_prime =df['CeFH'] * CeFH_hat;
    
    
    CeHF_hat = (pe + tb_mat[0]) ** (-epsilonD);
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeHF_hat = (pe) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeHF_hat = (pe + tb_mat[1] * tb_mat[0]) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    CeHF_prime = df['CeHF'] * CeHF_hat;
    
    
    CeFF_prime = df['CeFF'] * ((1 - jxbar_prime)/(1-df['jxbar'])) ** (1 + (1 - sigmastar)/theta) * pe ** (-epsilonDstar);
    
    ##
    VgHH = df['CeHH']/(1 - alpha);
    VgFF = df['CeFF']/(1 - alpha);
    
    VgFH = df['CeFH'] /(1 - alpha);
    # VgFH_prime = VgFH * pe ** ((1 - sigmastar) * (1 - alpha)) * (1 - (1 - jxbar_prime) ** (1 + (1 - sigmastar)/theta))/ (df['jxbar'] * (1 - df['jxbar']) ** ( (1-sigmastar)/theta));
    VgFH1_hat = (pe + tb_mat[0]) * CeFH1_hat;
    VgFH2_hat = pe**(1 - epsilonDstar) * ((1-j0_prime)**(1+(1-sigmastar)/theta) - (1-jxbar_prime)**(1+(1-sigmastar)/theta))/ (df['jxbar']  * (1 - df['jxbar'] )**( (1-sigmastar)/theta));
    VgFH1_prime = VgFH * VgFH1_hat;
    VgFH2_prime = VgFH * VgFH2_hat;
    VgFH_hat = VgFH1_hat + VgFH2_hat;
    if tax_scenario['Base']==1:
        VgFH_hat = pe * CeFH_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        VgFH_hat = (pe + tb_mat[0]) * CeFH_hat;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        VgFH_hat = (pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) * CeFH_hat;
    if np.isnan(VgFH_hat)==True:
            VgFH_hat=0
    VgFH_prime = VgFH * VgFH_hat;  

    VgHF = df['CeHF']/(1 - alpha);
    VgHF_hat = (pe + alpha * tb_mat[0]) * CeHF_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        VgHF_hat = pe * CeHF_hat;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        VgHF_hat = (pe + alpha * tb_mat[1] * tb_mat[0]) * CeHF_hat; 
    VgHF_prime = VgHF * VgHF_hat;
    Ge_prime = CeHH_prime + CeFH_prime;
    Ge_hat = Ge_prime/df['Ge'];
    Gestar_prime = CeFF_prime + CeHF_prime;
    Ce_prime = CeHH_prime + CeHF_prime;
    Ce_hat = Ce_prime/df['Ce'];
    Cestar_prime = CeFF_prime + CeFH_prime;
    Xe = df['Qe'] - df['Ge'];
    Xe_prime = Qe_prime - Ge_prime;
    Xg = VgFH - VgHF;
    Xg_prime = VgFH_prime - VgHF_prime;
    Xestar_prime = Qestar_prime - Gestar_prime;
    Qeworld_prime=Qe_prime+Qestar_prime;
    Geworld_prime=CeHH_prime+CeFH_prime+CeHF_prime+CeFF_prime;
    pai_g = VgFH - (pe + tb_mat[0]) * df['CeFH'] / (1 - alpha);
    subsidy_ratio = (tb_mat[0]/pe * (1 - alpha)) / (1 + tb_mat[0]/pe * (1 - alpha));
    
    Ve_prime=(pe+tb_mat[0]) * Ce_prime;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Ve_prime = (pe+tb_mat[0]) * CeHH_prime + pe * CeHF_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Ve_prime = (pe+tb_mat[0]) * CeHH_prime + (pe + tb_mat[1]*tb_mat[0]) * CeHF_prime;
    
    Vestar_prime= (pe+tb_mat[0]) * CeFH_prime  + pe * CeFF_prime;
    Vestar_prime= (pe+tb_mat[0]) * CeFH1_prime + pe * CeFH2_prime + pe * CeFF_prime;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid' :
        Vestar_prime = pe * Cestar_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Vestar_prime = (pe + tb_mat[0] - tb_mat[2]*tb_mat[0]) * CeFH_prime + pe * CeFF_prime;    
    
    Vg = df['Ce'] /(1-alpha);
    Vg_prime_hat = (pe + tb_mat[0]) * Ce_hat;
    Vg_prime = Vg * Vg_prime_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Vg_prime = CeHH_prime/(1-alpha) * (pe + tb_mat[0]) + CeHF_prime/(1-alpha) * pe; 
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Vg_prime = CeHH_prime/(1-alpha) * (pe + tb_mat[0]) + CeHF_prime/(1-alpha) * (pe + tb_mat[1] * tb_mat[0]);   
    
    Vgstar = df['Cestar'] /(1-alpha);
    Vgstar_prime = VgFH_prime + CeFF_prime/(1-alpha)* pe;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid' :
        Vgstar_prime = Cestar_prime/(1-alpha)* pe;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Vgstar_prime = CeFF_prime/(1-alpha)* pe + CeFH_prime/(1-alpha)* (pe + tb_mat[0] - tb_mat[2]*tb_mat[0]);
    
    Lg = alpha/(1-alpha) * df['Ge'];
    Lg_prime = alpha/(1-alpha) * (pe + tb_mat[0]) * Ge_prime;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid':
        Lg_prime = alpha/(1-alpha) * (pe + tb_mat[0]) * CeHH_prime + alpha/(1-alpha) * pe * CeFH_prime;  
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Lg_prime = alpha/(1-alpha) * (pe + tb_mat[0]) * CeHH_prime + alpha/(1-alpha) * (pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) * CeFH_prime;    
    
    Lgstar = alpha/(1-alpha) * df['Gestar'];
    Lgstar_prime = alpha/(1-alpha) * (pe+tb_mat[0]) * CeHF_prime +alpha/(1-alpha) * pe * CeFF_prime;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Lgstar_prime = alpha/(1-alpha) * pe * Gestar_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Lgstar_prime = alpha/(1-alpha) * (pe + tb_mat[1]*tb_mat[0]) * CeHF_prime + alpha/(1-alpha) * pe * CeFF_prime;

    delta_Le = (epsilonS/(epsilonS + 1)) * df['Qe'] * (petbte**(epsilonS + 1) - 1);
    delta_Lestar = (epsilonSstar/(epsilonSstar + 1)) * df['Qestar'] * (pe**(epsilonSstar + 1) - 1);
    
    def Func(a, beta, gamma):
        return (((1-gamma)*a**beta)/(1-gamma+gamma*a**beta)**2)
    if logit==1:
        delta_Le = beta * df['Qe'] * quad(Func,1,petbte, args=(beta, gamma))[0];
        delta_Lestar = beta * df['Qestar'] * quad(Func,1,pe, args=(beta, gamma))[0];
        
    leakage1 = -(Qestar_prime - df['Qestar'])/(Qeworld_prime - df['Qeworld']);
    leakage2 = -(Gestar_prime - df['Gestar'])/(Qeworld_prime - df['Qeworld']);
    leakage3 = -(Cestar_prime - df['Cestar'])/(Qeworld_prime - df['Qeworld']);   

    # if pe +  tb_mat[0]   <=0:
    #     print('varphi=')
    #     print(varphi)
    #     print('pe=')
    #     print(pe)
    #     print('tb=')
    #     print(tb_mat[0])
    #     tb_mat[0]=-pe+1
    # print('pe=')
    # print(pe)
    # print('tb=')
    # print(tb_mat[0])
    
    if pe<0:
        pe=tb_mat[0]
    test=math.log(pe)
    # print(test)
    # delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
    #        + sigma/(sigma-1)* (Vg_prime-Vg) + sigmastar/(sigmastar-1)* (Vgstar_prime-Vgstar) \
    #        - varphi * (Qeworld_prime - df['Qeworld']);
     
    delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
           + Vg * (alpha - 1) * math.log(pe + tb_mat[0]) + Vgstar * (1/theta) * math.log(df['jxbar']/j0_prime * (pe+tb_mat[0])**(-(1-alpha)*theta)) \
           - varphi * (Qeworld_prime - df['Qeworld']);

    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='purete' or tax_scenario['tax_sce']=='EC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
           + Vg * (alpha - 1) * math.log(pe + tb_mat[0]) + Vgstar * (alpha - 1) * math.log(pe) \
           - varphi * (Qeworld_prime - df['Qeworld']);

    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
               +  Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1/theta * math.log(df['jmbar']/jmbar_prime)) \
               +  Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1/theta * math.log(df['jxbar']/jxbar_prime)) \
               - varphi * (Qeworld_prime - df['Qeworld']);
    
    # print(tb_mat)
    # print(pe)
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
               +  Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1/theta * math.log(df['jmbar']/jmbar_prime)) \
               +  Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) + 1/theta * math.log(df['jxbar']/jxbar_prime)) \
               - varphi * (Qeworld_prime - df['Qeworld']);

    minuswelfare.welfare = delta_U/Vg;  
    ob=-minuswelfare.welfare;
    return ob
     

def minuswelfare_purete(te, tb_mat, varphi, tax_scenario, ParaList, df):
     #solve for equilibrium
    
    data = (tb_mat, te, varphi, tax_scenario, ParaList, df)
    pe = fsolve(equilibrium_hatalgebra,1,args=data)
    pe=pe[0]
    # print('pe='+str(pe))
    if pe-te<0 and tax_scenario['tax_sce']=='purete':
        pe=te;
        

        
    alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    # print(tb_mat)
     ## optimal values
    # jxbar_hat =   (1 - df['jxbar']) ** (-1) / (((1 - df['jxbar']) ** (-1) - 1) + (1 + (1 - alpha) * tb_mat[0]/pe) ** (-theta) * (1 + tb_mat[0]/pe) ** ((1 - alpha) * theta));
    jxbar_hat = pe**(-alpha*theta) * (pe+tb_mat[0])**(-(1-alpha)*theta) / ( df['jxbar'] * pe**(-alpha*theta) * (pe+tb_mat[0])**(-(1-alpha)*theta) + (1-df['jxbar']) * (pe + (1-alpha) * tb_mat[0])**(-theta));
    j0_hat = (pe+tb_mat[0])**(-(1-alpha)*theta) / (df['jxbar'] * (pe+tb_mat[0])**(-(1-alpha)*theta) + (1-df['jxbar']) * pe**(-(1-alpha)*theta) );
    jmbar_hat = 1 ;
    
    if tax_scenario['tax_sce']=='Unilateral':
        te=varphi;
        tb_mat[1]=1;
        
    if tax_scenario['tax_sce']=='purete':
        jxbar_hat = 1;   
        jmbar_hat = 1;

    
    if tax_scenario['tax_sce']=='puretc':
        te=tb_mat[0];
        jxbar_hat = 1;
        jmbar_hat = 1;
        tb_mat[1]=1;
    
    if tax_scenario['tax_sce']=='puretp':
        te=tb_mat[0];
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        tb_mat[1]=1;

    
    if tax_scenario['tax_sce']=='EC_hybrid':
        te=varphi;
        jxbar_hat = 1;
        jmbar_hat = 1;
        tb_mat[1]=1;

    
    if tax_scenario['tax_sce']=='PC_hybrid':
        te=tb_mat[0];
        if tb_mat[1]<0:
            tb_mat[1]=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[2] * tb_mat[0])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0])/(pe + tb_mat[1] * tb_mat[0])) ** (theta * (1 - alpha)));
    
    if tax_scenario['tax_sce']=='EP_hybrid':
        te=tb_mat[1];
        if tb_mat[1]<0:
            tb_mat[1]=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0]/pe) ** (theta * (1 - alpha)));
              
    if tax_scenario['tax_sce']=='EPC_hybrid':
        te=varphi;
        if tb_mat[1]<0:
            tb_mat[1]=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[2] * tb_mat[0])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0])/(pe + tb_mat[1] * tb_mat[0])) ** (theta * (1 - alpha)));
    
    jxbar_prime = jxbar_hat * df['jxbar'];
    jmbar_prime = jmbar_hat * df['jmbar'];
    j0_prime = j0_hat * df['jxbar'];

    def tempFunction(i, theta, sigmastar):
        return (i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1)) 
    
    Bfunvec1_prime = quad(tempFunction,0,j0_prime, args=(theta, sigmastar))[0];
    Bfunvec2_prime = quad(tempFunction,0,jxbar_prime, args=(theta, sigmastar))[0];

    #if te is too large, Home stop producing
    petbte = pe + tb_mat[0] - te;
    z = pe + tb_mat[0] >= te;
    petbte = petbte * z;

    Qe_hat = (petbte) ** epsilonS;
    Qestar_hat = pe ** epsilonSstar;
          
    if logit==1:
        epsilonS=beta*(1-gamma)/(1-gamma+gamma*petbte**beta);
        epsilonSstar=beta*(1-gamma)/(1-gamma+gamma*pe**beta);
        Qe_hat = (petbte)**beta/(1-gamma+gamma*(petbte)**beta);
        Qestar_hat = pe**beta/(1-gamma+gamma*pe**beta);
 
    Qe_prime = df['Qe'] * Qe_hat;
    Qestar_prime = df['Qestar'] * Qestar_hat;
 
    CeHH_hat = (pe + tb_mat[0]) ** (-epsilonD) * jmbar_hat ** (1 + (1 - sigma)/theta);
    CeHH_prime = df['CeHH'] * CeHH_hat;
       
    
    # CeFH_hat = (1 + (1 - sigmastar)/theta) * pe ** (-(1 - alpha) * sigmastar) * (pe + tb_mat[0]) ** (-alpha) * Bfunvec_prime/(df['jxbar'] ** (1 +1/theta)) * (1 - df['jxbar']) ** (sigmastar/theta);
    CeFH1_hat = (pe +tb_mat[0])**(-epsilonDstar) * j0_hat**(1 + (1 - sigmastar)/theta);
    CeFH2_hat = (1 + (1 - sigmastar)/theta) * ((1-df['jxbar'])/df['jxbar'])**(sigmastar/theta) * pe**(-epsilonDstar) * (1 + tb_mat[0]/pe)**(-alpha) * (Bfunvec2_prime - Bfunvec1_prime)/df['jxbar']**(1+(1-sigmastar)/theta);
    CeFH1_prime = df['CeFH'] * CeFH1_hat;
    CeFH2_prime = df['CeFH'] * CeFH2_hat;
    CeFH_hat = CeFH1_hat + CeFH2_hat;
    if tax_scenario['Base']==1:
        CeFH_hat = pe ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeFH_hat = (pe + tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeFH_hat = (pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if np.isnan(CeFH_hat)==True:
        CeFH_hat=0
    CeFH_prime =df['CeFH'] * CeFH_hat;
    
    
    CeHF_hat = (pe + tb_mat[0]) ** (-epsilonD);
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeHF_hat = (pe) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeHF_hat = (pe + tb_mat[1] * tb_mat[0]) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    CeHF_prime = df['CeHF'] * CeHF_hat;
    
    
    CeFF_prime = df['CeFF'] * ((1 - jxbar_prime)/(1-df['jxbar'])) ** (1 + (1 - sigmastar)/theta) * pe ** (-epsilonDstar);
    
    ##
    VgHH = df['CeHH']/(1 - alpha);
    VgFF = df['CeFF']/(1 - alpha);
    
    VgFH = df['CeFH'] /(1 - alpha);
    # VgFH_prime = VgFH * pe ** ((1 - sigmastar) * (1 - alpha)) * (1 - (1 - jxbar_prime) ** (1 + (1 - sigmastar)/theta))/ (df['jxbar'] * (1 - df['jxbar']) ** ( (1-sigmastar)/theta));
    VgFH1_hat = (pe + tb_mat[0]) * CeFH1_hat;
    VgFH2_hat = pe**(1 - epsilonDstar) * ((1-j0_prime)**(1+(1-sigmastar)/theta) - (1-jxbar_prime)**(1+(1-sigmastar)/theta))/ (df['jxbar']  * (1 - df['jxbar'] )**( (1-sigmastar)/theta));
    VgFH1_prime = VgFH * VgFH1_hat;
    VgFH2_prime = VgFH * VgFH2_hat;
    VgFH_hat = VgFH1_hat + VgFH2_hat;
    if tax_scenario['Base']==1:
        VgFH_hat = pe * CeFH_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        VgFH_hat = (pe + tb_mat[0]) * CeFH_hat;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        VgFH_hat = (pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) * CeFH_hat;
    if np.isnan(VgFH_hat)==True:
            VgFH_hat=0
    VgFH_prime = VgFH * VgFH_hat;  

    VgHF = df['CeHF']/(1 - alpha);
    VgHF_hat = (pe + alpha * tb_mat[0]) * CeHF_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        VgHF_hat = pe * CeHF_hat;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        VgHF_hat = (pe + alpha * tb_mat[1] * tb_mat[0]) * CeHF_hat; 
    VgHF_prime = VgHF * VgHF_hat;
    Ge_prime = CeHH_prime + CeFH_prime;
    Ge_hat = Ge_prime/df['Ge'];
    Gestar_prime = CeFF_prime + CeHF_prime;
    Ce_prime = CeHH_prime + CeHF_prime;
    Ce_hat = Ce_prime/df['Ce'];
    Cestar_prime = CeFF_prime + CeFH_prime;
    Xe = df['Qe'] - df['Ge'];
    Xe_prime = Qe_prime - Ge_prime;
    Xg = VgFH - VgHF;
    Xg_prime = VgFH_prime - VgHF_prime;
    Xestar_prime = Qestar_prime - Gestar_prime;
    Qeworld_prime=Qe_prime+Qestar_prime;
    Geworld_prime=CeHH_prime+CeFH_prime+CeHF_prime+CeFF_prime;
    pai_g = VgFH - (pe + tb_mat[0]) * df['CeFH'] / (1 - alpha);
    subsidy_ratio = (tb_mat[0]/pe * (1 - alpha)) / (1 + tb_mat[0]/pe * (1 - alpha));
    
    Ve_prime=(pe+tb_mat[0]) * Ce_prime;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Ve_prime = (pe+tb_mat[0]) * CeHH_prime + pe * CeHF_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Ve_prime = (pe+tb_mat[0]) * CeHH_prime + (pe + tb_mat[1]*tb_mat[0]) * CeHF_prime;
    
    Vestar_prime= (pe+tb_mat[0]) * CeFH_prime  + pe * CeFF_prime;
    Vestar_prime= (pe+tb_mat[0]) * CeFH1_prime + pe * CeFH2_prime + pe * CeFF_prime;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid' :
        Vestar_prime = pe * Cestar_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
       Vestar_prime = (pe + tb_mat[0] - tb_mat[2]*tb_mat[0]) * CeFH_prime + pe * CeFF_prime;    
    
    Vg = df['Ce'] /(1-alpha);
    Vg_prime_hat = (pe + tb_mat[0]) * Ce_hat;
    Vg_prime = Vg * Vg_prime_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Vg_prime = CeHH_prime/(1-alpha) * (pe + tb_mat[0]) + CeHF_prime/(1-alpha) * pe; 
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Vg_prime = CeHH_prime/(1-alpha) * (pe + tb_mat[0]) + CeHF_prime/(1-alpha) * (pe + tb_mat[1] * tb_mat[0]);   
    
    Vgstar = df['Cestar'] /(1-alpha);
    Vgstar_prime = VgFH_prime + CeFF_prime/(1-alpha)* pe;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid' :
        Vgstar_prime = Cestar_prime/(1-alpha)* pe;
    if tax_scenario['tax_sce']=='PC_hybrid':
        Vgstar_prime = CeFF_prime/(1-alpha)* pe + CeFH_prime/(1-alpha)* (pe + tb_mat[0] - tb_mat[2]*tb_mat[0]);
    
    Lg = alpha/(1-alpha) * df['Ge'];
    Lg_prime = alpha/(1-alpha) * (pe + tb_mat[0]) * Ge_prime;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid':
        Lg_prime = alpha/(1-alpha) * (pe + tb_mat[0]) * CeHH_prime + alpha/(1-alpha) * pe * CeFH_prime;  
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Lg_prime = alpha/(1-alpha) * (pe + tb_mat[0]) * CeHH_prime + alpha/(1-alpha) * (pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) * CeFH_prime;    
    
    Lgstar = alpha/(1-alpha) * df['Gestar'];
    Lgstar_prime = alpha/(1-alpha) * (pe+tb_mat[0]) * CeHF_prime +alpha/(1-alpha) * pe * CeFF_prime;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Lgstar_prime = alpha/(1-alpha) * pe * Gestar_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Lgstar_prime = alpha/(1-alpha) * (pe + tb_mat[1]*tb_mat[0]) * CeHF_prime + alpha/(1-alpha) * pe * CeFF_prime;

    delta_Le = (epsilonS/(epsilonS + 1)) * df['Qe'] * (petbte**(epsilonS + 1) - 1);
    delta_Lestar = (epsilonSstar/(epsilonSstar + 1)) * df['Qestar'] * (pe**(epsilonSstar + 1) - 1);
    
    def Func(a, beta, gamma):
        return (((1-gamma)*a**beta)/(1-gamma+gamma*a**beta)**2)
    if logit==1:
        delta_Le = beta * df['Qe'] * quad(Func,1,petbte, args=(beta, gamma))[0];
        delta_Lestar = beta * df['Qestar'] * quad(Func,1,pe, args=(beta, gamma))[0];
 
    
    leakage1 = -(Qestar_prime - df['Qestar'])/(Qeworld_prime - df['Qeworld']);
    leakage2 = -(Gestar_prime - df['Gestar'])/(Qeworld_prime - df['Qeworld']);
    leakage3 = -(Cestar_prime - df['Cestar'])/(Qeworld_prime - df['Qeworld']);   


    # delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
    #        + sigma/(sigma-1)* (Vg_prime-Vg) + sigmastar/(sigmastar-1)* (Vgstar_prime-Vgstar) \
    #        - varphi * (Qeworld_prime - df['Qeworld']);
       
    delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
           + Vg * (alpha - 1) * math.log(pe + tb_mat[0]) + Vgstar * (1/theta) * math.log(df['jxbar']/j0_prime * (pe+tb_mat[0])**(-(1-alpha)*theta)) \
           - varphi * (Qeworld_prime - df['Qeworld']);

    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='purete' or tax_scenario['tax_sce']=='EC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
           + Vg * (alpha - 1) * math.log(pe +  tb_mat[0]) + Vgstar * (alpha - 1) * math.log(pe) \
           - varphi * (Qeworld_prime - df['Qeworld']);

    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
               +  Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1/theta * math.log(df['jmbar']/jmbar_prime)) \
               +  Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1/theta * math.log(df['jxbar']/jxbar_prime)) \
               - varphi * (Qeworld_prime - df['Qeworld']);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
               +  Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1/theta * math.log(df['jmbar']/jmbar_prime)) \
               +  Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0] - tb_mat[2] * tb_mat[0]) + 1/theta * math.log(df['jxbar']/jxbar_prime)) \
               - varphi * (Qeworld_prime - df['Qeworld']);

    minuswelfare.welfare = delta_U/Vg;  
    ob=-minuswelfare.welfare;
    return ob
          

def minimization(df, tb_mat, te, varphi, tax_scenario, ParaList):
    alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList

    
    if tax_scenario['tax_sce']=='Baseline':
        tb=0
        prop=1
        te=0
        return pd.Series({'tb': tb, 'prop': prop, 'te': te})
    elif tax_scenario['tax_sce']=='PC_hybrid':
        res = minimize(minuswelfare, [0, 1, 1], bounds=[(0,None),(0,1),(0, 1)],method='trust-constr',args=(te, varphi, tax_scenario, ParaList, df));
        tb_mat = res.x;
        tb=tb_mat[0];
        prop=tb_mat[1];   
        prop2=tb_mat[2];
        te=tb
        return pd.Series({'tb': tb, 'prop': prop, 'prop2': prop2,'te': te})
    
    elif tax_scenario['tax_sce']=='EPC_hybrid':
        res = minimize(minuswelfare, [0, 1, 1], bounds=[(0,None),(0,1),(0, 1)],method='trust-constr',args=(te, varphi, tax_scenario, ParaList, df));
        tb_mat = res.x;
        tb=tb_mat[0];
        prop=tb_mat[1];   
        prop2=tb_mat[2];
        te=varphi
        return pd.Series({'tb': tb, 'prop': prop, 'prop2': prop2,'te': te})
  
    elif tax_scenario['tax_sce']=='EP_hybrid':
        res = minimize(minuswelfare, [0, 1], bounds=[(0,None),(0,None)],method='L-BFGS-B', args=(te, varphi, tax_scenario, ParaList, df));
        tb_mat = res.x;
        tb=tb_mat[0];
        te=tb_mat[1];   
        prop=te-tb;
        return pd.Series({'tb': tb, 'prop': prop, 'te': te})
    elif tax_scenario['tax_sce']=='Unilateral' or tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EC_hybrid' :
        res = minimize(minuswelfare, [0, 1], method='nelder-mead', args=(te, varphi, tax_scenario, ParaList, df));
        tb_mat = res.x;
        tb=tb_mat[0];
        prop=tb_mat[1];
        print("tax returned is", tb_mat, "flag is", res.success, "tax scenario is", tax_scenario['tax_sce'])
        if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='puretp':
            te=tb
        elif tax_scenario['tax_sce']=='Unilateral' or tax_scenario['tax_sce']=='EC_hybrid':
            te=varphi
        return pd.Series({'tb': tb, 'prop': prop, 'te': te})
    elif tax_scenario['tax_sce']=='purete':
        res = minimize(minuswelfare_purete, 0, method='nelder-mead', args=(tb_mat, varphi, tax_scenario, ParaList, df));
        te = res.x[0]
        tb_mat = [0,1];
        tb=tb_mat[0];
        prop=tb_mat[1];  
        return pd.Series({'tb': tb, 'prop': prop, 'te': te})
   
    
def callback_pe(pe, df, varphi, tax_scenario, ParaList):
    alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList

    ## optimal values
    # jxbar_hat =   (1 - df['jxbar']) ** (-1) / (((1 - df['jxbar']) ** (-1) - 1) + (1 + (1 - alpha) * df['tb']/pe) ** (-theta) * (1 + df['tb']/pe) ** ((1 - alpha) * theta));
    jxbar_hat = pe**(-alpha*theta) * (pe+df['tb'])**(-(1-alpha)*theta) / ( df['jxbar'] * pe**(-alpha*theta) * (pe+df['tb'])**(-(1-alpha)*theta) + (1-df['jxbar']) * (pe + (1-alpha) * df['tb'])**(-theta));
    j0_hat = (pe+df['tb'])**(-(1-alpha)*theta) / (df['jxbar'] * (pe+df['tb'])**(-(1-alpha)*theta) + (1-df['jxbar']) * pe**(-(1-alpha)*theta) );
    jmbar_hat = 1 ;
    
    if tax_scenario['tax_sce']=='Unilateral':
        df['te']=varphi;
        df['prop']=1;
        
    if tax_scenario['tax_sce']=='purete':
        jxbar_hat = 1;   
        jmbar_hat = 1;

    if tax_scenario['tax_sce']=='puretc':
        df['te']=df['tb'];
        jxbar_hat = 1;
        jmbar_hat = 1;
        df['prop']=1;
    
    if tax_scenario['tax_sce']=='puretp':
        df['te']=df['tb'];
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + df['tb']/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + df['tb']/pe) ** (theta * (1 - alpha)));
        df['prop']=1;

    
    if tax_scenario['tax_sce']=='EC_hybrid':
        df['te']=varphi;
        jxbar_hat = 1;
        jmbar_hat = 1;
        df['prop']=1;

    
    if tax_scenario['tax_sce']=='PC_hybrid':
        df['te']=df['tb'];
        if df['prop']<0:
            df['prop']=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (df['tb'] - df['prop2'] * df['tb'])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + df['tb'])/(pe + df['prop'] * df['tb'])) ** (theta * (1 - alpha)));
    
    if tax_scenario['tax_sce']=='EP_hybrid':
        # df['te']=df['tb']+df['prop'];
        # if df['prop']<0:
        #     df['prop']=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + df['tb']/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + df['tb']/pe) ** (theta * (1 - alpha)));
  
    if tax_scenario['tax_sce']=='EPC_hybrid':
        df['te']=varphi;
        if df['prop']<0:
            df['prop']=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (df['tb'] - df['prop2'] * df['tb'])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + df['tb'])/(pe + df['prop'] * df['tb'])) ** (theta * (1 - alpha)));
              
    
    jxbar_prime = jxbar_hat * df['jxbar'];
    jmbar_prime = jmbar_hat * df['jmbar'];
    j0_prime = j0_hat * df['jxbar'];

    def tempFunction(i, theta, sigmastar):
        return (i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1)) 
    
    Bfunvec1_prime = quad(tempFunction,0,j0_prime, args=(theta, sigmastar))[0];
    Bfunvec2_prime = quad(tempFunction,0,jxbar_prime, args=(theta, sigmastar))[0];

    #if te is too large, HoGe stop producing
    petbte = pe + df['tb'] - df['te'];
    z = pe +  df['tb'] >= df['te'];
    petbte = petbte * z;
    
    Qe_hat = (petbte) ** epsilonS;
    Qestar_hat = pe ** epsilonSstar;
          
    if logit==1:
        epsilonS=beta*(1-gamma)/(1-gamma+gamma*petbte**beta);
        epsilonSstar=beta*(1-gamma)/(1-gamma+gamma*pe**beta);
        Qe_hat = (petbte)**beta/(1-gamma+gamma*(petbte)**beta);
        Qestar_hat = pe**beta/(1-gamma+gamma*pe**beta);
 
    Qe_prime = df['Qe'] * Qe_hat;
    Qestar_prime = df['Qestar'] * Qestar_hat;
    
    CeHH_hat = (pe + df['tb']) ** (-epsilonD) * jmbar_hat ** (1 + (1 - sigma)/theta);
    CeHH_prime = df['CeHH'] * CeHH_hat;
       
    
    # CeFH_hat = (1 + (1 - sigmastar)/theta) * pe ** (-(1 - alpha) * sigmastar) * (pe + df['tb']) ** (-alpha) * Bfunvec_prime/(df['jxbar'] ** (1 +1/theta)) * (1 - df['jxbar']) ** (sigmastar/theta);
    CeFH1_hat = (pe +df['tb'])**(-epsilonDstar) * j0_hat**(1 + (1 - sigmastar)/theta);
    CeFH2_hat = (1 + (1 - sigmastar)/theta) * ((1-df['jxbar'])/df['jxbar'])**(sigmastar/theta) * pe**(-epsilonDstar) * (1 + df['tb']/pe)**(-alpha) * (Bfunvec2_prime - Bfunvec1_prime)/df['jxbar']**(1+(1-sigmastar)/theta);
    CeFH1_prime = df['CeFH'] * CeFH1_hat;
    CeFH2_prime = df['CeFH'] * CeFH2_hat;
    CeFH_hat = CeFH1_hat + CeFH2_hat;
    if tax_scenario['Base']==1:
        CeFH_hat = pe ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeFH_hat = (pe + df['tb']) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeFH_hat = (pe + df['tb'] - df['prop2'] * df['tb']) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if np.isnan(CeFH_hat)==True:
        CeFH_hat=0
    CeFH_prime =df['CeFH'] * CeFH_hat;
    
    
    CeHF_hat = (pe + df['tb']) ** (-epsilonD);
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeHF_hat = (pe) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeHF_hat = (pe + df['prop'] * df['tb']) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    CeHF_prime = df['CeHF'] * CeHF_hat;
    
    
    CeFF_prime = df['CeFF'] * ((1 - jxbar_prime)/(1-df['jxbar'])) ** (1 + (1 - sigmastar)/theta) * pe ** (-epsilonDstar);
    
    diff = Qe_prime + Qestar_prime - (CeHH_prime + CeFH_prime + CeHF_prime + CeFF_prime);
    #print('diff='+str(diff))
    return diff
    
    
def callback(df, varphi, tax_scenario, ParaList):
    #solve for equilibrium
    if df['prop']<0:
        df['prop']=0
        
    pe = fsolve(callback_pe,1,args=(df, varphi, tax_scenario, ParaList))
    pe=pe[0]
    # print('pe='+str(pe))
    alpha, theta, sigma, sigmastar, epsilonD,epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    
   ## optimal values
    # jxbar_hat =   (1 - df['jxbar']) ** (-1) / (((1 - df['jxbar']) ** (-1) - 1) + (1 + (1 - alpha) * df['tb']/pe) ** (-theta) * (1 + df['tb']/pe) ** ((1 - alpha) * theta));
    jxbar_hat = pe**(-alpha*theta) * (pe+df['tb'])**(-(1-alpha)*theta) / ( df['jxbar'] * pe**(-alpha*theta) * (pe+df['tb'])**(-(1-alpha)*theta) + (1-df['jxbar']) * (pe + (1-alpha) * df['tb'])**(-theta));
    j0_hat = (pe+df['tb'])**(-(1-alpha)*theta) / (df['jxbar'] * (pe+df['tb'])**(-(1-alpha)*theta) + (1-df['jxbar']) * pe**(-(1-alpha)*theta) );
    jmbar_hat = 1 ;
    
    if tax_scenario['tax_sce']=='Unilateral':
        df['te']=varphi;
        df['prop']=1;
        
    if tax_scenario['tax_sce']=='purete':
        jxbar_hat = 1;   
        jmbar_hat = 1;

    if tax_scenario['tax_sce']=='puretc':
        df['te']=df['tb'];
        jxbar_hat = 1;
        jmbar_hat = 1;
        df['prop']=1;
    
    if tax_scenario['tax_sce']=='puretp':
        df['te']=df['tb'];
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + df['tb']/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + df['tb']/pe) ** (theta * (1 - alpha)));
        df['prop']=1;

    
    if tax_scenario['tax_sce']=='EC_hybrid':
        df['te']=varphi;
        jxbar_hat = 1;
        jmbar_hat = 1;
        df['prop']=1;

    
    if tax_scenario['tax_sce']=='PC_hybrid':
        df['te']=df['tb'];
        if df['prop']<0:
            df['prop']=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (df['tb'] - df['prop2'] * df['tb'])/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + df['tb'])/(pe + df['prop'] * df['tb'])) ** (theta * (1 - alpha)));
    
    if tax_scenario['tax_sce']=='EP_hybrid':
        # df['te']=df['tb']+df['prop'];
        # if df['prop']<0:
        #     df['prop']=0
        jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + df['tb']/pe) ** (theta * (1 - alpha)));
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + df['tb']/pe) ** (theta * (1 - alpha)));
              
    if tax_scenario['tax_sce']=='EPC_hybrid':
       df['te']=varphi;
       if df['prop']<0:
            df['prop']=0
       jxbar_hat =  (1 - df['jxbar']) ** (-1) / ((1 - df['jxbar']) ** (-1) - 1 + (1 + (df['tb'] - df['prop2'] * df['tb'])/pe) ** (theta * (1 - alpha)));
       jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + df['tb'])/(pe + df['prop'] * df['tb'])) ** (theta * (1 - alpha)));
    
    jxbar_prime = jxbar_hat * df['jxbar'];
    jmbar_prime = jmbar_hat * df['jmbar'];
    j0_prime = j0_hat * df['jxbar'];

    def tempFunction(i, theta, sigmastar):
        return (i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1)) 
    
    Bfunvec1_prime = quad(tempFunction,0,j0_prime, args=(theta, sigmastar))[0];
    Bfunvec2_prime = quad(tempFunction,0,jxbar_prime, args=(theta, sigmastar))[0];

    #if te is too large, HoGe stop producing
    petbte = pe + df['tb'] - df['te'];
    z = pe +  df['tb'] >= df['te'];

    petbte = petbte * z;

    Qe_hat = (petbte) ** epsilonS;
    Qestar_hat = pe ** epsilonSstar;
          
    if logit==1:
        epsilonS=beta*(1-gamma)/(1-gamma+gamma*petbte**beta);
        epsilonSstar=beta*(1-gamma)/(1-gamma+gamma*pe**beta);
        Qe_hat = (petbte)**beta/(1-gamma+gamma*(petbte)**beta);
        Qestar_hat = pe**beta/(1-gamma+gamma*pe**beta);
 
    Qe_prime = df['Qe'] * Qe_hat;
    Qestar_prime = df['Qestar'] * Qestar_hat;
    
    
    CeHH_hat = (pe + df['tb']) ** (-epsilonD) * jmbar_hat ** (1 + (1 - sigma)/theta);
    CeHH_prime = df['CeHH'] * CeHH_hat;
       
    
    # CeFH_hat = (1 + (1 - sigmastar)/theta) * pe ** (-(1 - alpha) * sigmastar) * (pe + df['tb']) ** (-alpha) * Bfunvec_prime/(df['jxbar'] ** (1 +1/theta)) * (1 - df['jxbar']) ** (sigmastar/theta);
    CeFH1_hat = (pe +df['tb'])**(-epsilonDstar) * j0_hat**(1 + (1 - sigmastar)/theta);
    CeFH2_hat = (1 + (1 - sigmastar)/theta) * ((1-df['jxbar'])/df['jxbar'])**(sigmastar/theta) * pe**(-epsilonDstar) * (1 + df['tb']/pe)**(-alpha) * (Bfunvec2_prime - Bfunvec1_prime)/df['jxbar']**(1+(1-sigmastar)/theta);
    CeFH1_prime = df['CeFH'] * CeFH1_hat;
    CeFH2_prime = df['CeFH'] * CeFH2_hat;
    CeFH_hat = CeFH1_hat + CeFH2_hat;
    if tax_scenario['Base']==1:
        CeFH_hat = pe ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeFH_hat = (pe + df['tb']) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeFH_hat = (pe + df['tb'] - df['prop2'] * df['tb']) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar)/theta);
    
    if np.isnan(CeFH_hat)==True:
        CeFH_hat=0
    CeFH_prime =df['CeFH'] * CeFH_hat;
    
    
    CeHF_hat = (pe + df['tb']) ** (-epsilonD);
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        CeHF_hat = (pe) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        CeHF_hat = (pe + df['prop'] * df['tb']) ** (-epsilonD) * ((1 - jmbar_prime)/(1 - df['jmbar'])) ** (1 + (1 - sigma)/theta);
    
    CeHF_prime = df['CeHF'] * CeHF_hat;
    
    
    CeFF_prime = df['CeFF'] * ((1 - jxbar_prime)/(1-df['jxbar'])) ** (1 + (1 - sigmastar)/theta) * pe ** (-epsilonDstar);
    
    ##
    VgHH = df['CeHH']/(1 - alpha);
    VgFF = df['CeFF']/(1 - alpha);
    
    VgFH = df['CeFH'] /(1 - alpha);
    # VgFH_prime = VgFH * pe ** ((1 - sigmastar) * (1 - alpha)) * (1 - (1 - jxbar_prime) ** (1 + (1 - sigmastar)/theta))/ (df['jxbar'] * (1 - df['jxbar']) ** ( (1-sigmastar)/theta));
    VgFH1_hat = (pe + df['tb']) * CeFH1_hat;
    VgFH2_hat = pe**(1 - epsilonDstar) * ((1-j0_prime)**(1+(1-sigmastar)/theta) - (1-jxbar_prime)**(1+(1-sigmastar)/theta))/ (df['jxbar']  * (1 - df['jxbar'] )**( (1-sigmastar)/theta));
    VgFH1_prime = VgFH * VgFH1_hat;
    VgFH2_prime = VgFH * VgFH2_hat;
    VgFH_hat = VgFH1_hat + VgFH2_hat;
    if tax_scenario['Base']==1:
        VgFH_hat = pe * CeFH_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        VgFH_hat = (pe + df['tb']) * CeFH_hat;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        VgFH_hat = (pe + df['tb'] - df['prop2'] * df['tb']) * CeFH_hat;
    if np.isnan(VgFH_hat)==True:
            VgFH_hat=0
    VgFH_prime = VgFH * VgFH_hat;  

    VgHF = df['CeHF']/(1 - alpha);
    VgHF_hat = (pe + alpha * df['tb']) * CeHF_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        VgHF_hat = pe * CeHF_hat;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        VgHF_hat = (pe + alpha * df['prop'] * df['tb']) * CeHF_hat; 
    VgHF_prime = VgHF * VgHF_hat;
    Ge_prime = CeHH_prime + CeFH_prime;
    Ge_hat = Ge_prime/df['Ge'];
    Gestar_prime = CeFF_prime + CeHF_prime;
    Ce_prime = CeHH_prime + CeHF_prime;
    Ce_hat = Ce_prime/df['Ce'];
    Cestar_prime = CeFF_prime + CeFH_prime;
    Xe = df['Qe'] - df['Ge'];
    Xe_prime = Qe_prime - Ge_prime;
    Xg = VgFH - VgHF;
    Xg_prime = VgFH_prime - VgHF_prime;
    Xestar_prime = Qestar_prime - Gestar_prime;
    Qeworld_prime=Qe_prime+Qestar_prime;
    Geworld_prime=CeHH_prime+CeFH_prime+CeHF_prime+CeFF_prime;
    pai_g = VgFH - (pe + df['tb']) * df['CeFH'] / (1 - alpha);
    subsidy_ratio = (df['tb']/pe * (1 - alpha)) / (1 + df['tb']/pe * (1 - alpha));
    
    Ve_prime=(pe+df['tb']) * Ce_prime;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Ve_prime = (pe+df['tb']) * CeHH_prime + pe * CeHF_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Ve_prime = (pe+df['tb']) * CeHH_prime + (pe + df['prop']*df['tb']) * CeHF_prime;
    
    Vestar_prime= (pe+df['tb']) * CeFH_prime  + pe * CeFF_prime;
    Vestar_prime= (pe+df['tb']) * CeFH1_prime + pe * CeFH2_prime + pe * CeFF_prime;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid' :
        Vestar_prime = pe * Cestar_prime;
    if tax_scenario['tax_sce']=='PC_hybrid':
       Vestar_prime = (pe + df['tb'] - df['prop2']*df['tb']) * CeFH_prime + pe * CeFF_prime;    
    
    Vg = df['Ce'] /(1-alpha);
    Vg_prime_hat = (pe + df['tb']) * Ce_hat;
    Vg_prime = Vg * Vg_prime_hat;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Vg_prime = CeHH_prime/(1-alpha) * (pe + df['tb']) + CeHF_prime/(1-alpha) * pe; 
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Vg_prime = CeHH_prime/(1-alpha) * (pe + df['tb']) + CeHF_prime/(1-alpha) * (pe + df['prop'] * df['tb']);   
    
    Vgstar = df['Cestar'] /(1-alpha);
    Vgstar_prime = VgFH_prime + CeFF_prime/(1-alpha)* pe;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid' :
        Vgstar_prime = Cestar_prime/(1-alpha)* pe;
    if tax_scenario['tax_sce']=='PC_hybrid':
        Vgstar_prime = CeFF_prime/(1-alpha)* pe + CeFH_prime/(1-alpha)* (pe + df['tb'] - df['prop2']*df['tb']);
    
    Lg = alpha/(1-alpha) * df['Ge'];
    Lg_prime = alpha/(1-alpha) * (pe + df['tb']) * Ge_prime;
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='EC_hybrid':
        Lg_prime = alpha/(1-alpha) * (pe + df['tb']) * CeHH_prime + alpha/(1-alpha) * pe * CeFH_prime;  
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Lg_prime = alpha/(1-alpha) * (pe + df['tb']) * CeHH_prime + alpha/(1-alpha) * (pe + df['tb'] - df['prop2'] * df['tb']) * CeFH_prime;    
    
    Lgstar = alpha/(1-alpha) * df['Gestar'];
    Lgstar_prime = alpha/(1-alpha) * (pe+df['tb']) * CeHF_prime +alpha/(1-alpha) * pe * CeFF_prime;
    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        Lgstar_prime = alpha/(1-alpha) * pe * Gestar_prime;
    if tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        Lgstar_prime = alpha/(1-alpha) * (pe + df['prop']*df['tb']) * CeHF_prime + alpha/(1-alpha) * pe * CeFF_prime;

    delta_Le = (epsilonS/(epsilonS + 1)) * df['Qe'] * (petbte**(epsilonS + 1) - 1);
    delta_Lestar = (epsilonSstar/(epsilonSstar + 1)) * df['Qestar'] * (pe**(epsilonSstar + 1) - 1);
    
    def Func(a, beta, gamma):
        return (((1-gamma)*a**beta)/(1-gamma+gamma*a**beta)**2)
    if logit==1:
        delta_Le = beta * df['Qe'] * quad(Func,1,petbte, args=(beta, gamma))[0];
        delta_Lestar = beta * df['Qestar'] * quad(Func,1,pe, args=(beta, gamma))[0];
    
    leakage1 = -(Qestar_prime - df['Qestar'])/(Qeworld_prime - df['Qeworld']);
    leakage2 = -(Gestar_prime - df['Gestar'])/(Qeworld_prime - df['Qeworld']);
    leakage3 = -(Cestar_prime - df['Cestar'])/(Qeworld_prime - df['Qeworld']);   
    chg_extraction=Qestar_prime-df['Qestar'];
    chg_production=Gestar_prime-df['Gestar'];
    chg_consumption=Cestar_prime-df['Cestar'];
    chg_Qeworld=Qeworld_prime-df['Qeworld'];

    # delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
    #        + sigma/(sigma-1)* (Vg_prime-Vg) + sigmastar/(sigmastar-1)* (Vgstar_prime-Vgstar) \
    #        - varphi * (Qeworld_prime - df['Qeworld']);
       
    delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
           + Vg * (alpha - 1) * math.log(pe + df['tb']) + Vgstar * (1/theta) * math.log(df['jxbar']/j0_prime * (pe+df['tb'])**(-(1-alpha)*theta)) \
           - varphi * (Qeworld_prime - df['Qeworld']);
    # print("pe")
    # print(pe)
    # print("tb")
    # print(df['tb'])
    if pe<0:
        pe=0.0001
    if tax_scenario['tax_sce']=='puretc' or tax_scenario['tax_sce']=='purete' or tax_scenario['tax_sce']=='EC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
           + Vg * (alpha - 1) * math.log(pe +  df['tb']) + Vgstar * (alpha - 1) * math.log(pe) \
           - varphi * (Qeworld_prime - df['Qeworld']);

    if tax_scenario['tax_sce']=='puretp' or tax_scenario['tax_sce']=='EP_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
               +  Vg * ((alpha - 1) * math.log(pe + df['tb']) + 1/theta * math.log(df['jmbar']/jmbar_prime)) \
               +  Vgstar * ((alpha - 1) * math.log(pe + df['tb']) + 1/theta * math.log(df['jxbar']/jxbar_prime)) \
               - varphi * (Qeworld_prime - df['Qeworld']);

    if  tax_scenario['tax_sce']=='PC_hybrid' or tax_scenario['tax_sce']=='EPC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
               +  Vg * ((alpha - 1) * math.log(pe + df['tb']) + 1/theta * math.log(df['jmbar']/jmbar_prime)) \
               +  Vgstar * ((alpha - 1) * math.log(pe + df['tb'] - df['prop2']*df['tb']) + 1/theta * math.log(df['jxbar']/jxbar_prime)) \
               -  varphi * (Qeworld_prime - df['Qeworld']);
    
    
    welfare = delta_U/Vg*100;  
    welfare_noexternality = (delta_U + varphi * (Qeworld_prime - df['Qeworld']) )/Vg*100;  
    
    return(pd.Series({ 'varphi': varphi, 'pe': pe, 'jxbar_prime': jxbar_prime, 'jmbar_prime': jmbar_prime, 'j0_prime': j0_prime, \
                      'Qe_prime': Qe_prime, 'Qestar_prime': Qestar_prime, 'Qeworld_prime': Qeworld_prime, \
                      'CeHH_prime': CeHH_prime,'CeFH_prime': CeFH_prime, 'CeHF_prime': CeHF_prime, 'CeFF_prime': CeFF_prime,\
                      'Ge_prime': Ge_prime,'Ce_prime': Ce_prime, 'Gestar_prime': Gestar_prime,'Cestar_prime': Cestar_prime, \
                      'VgFH_prime': VgFH_prime, 'VgHF_prime': VgHF_prime, \
                      'VgFH1_prime': VgFH1_prime, 'VgFH2_prime': VgFH2_prime, \
                      'CeFH1_prime': CeFH1_prime, 'CeFH2_prime': CeFH2_prime, \
                      'Vg_prime': Vg_prime, 'Vgstar_prime': Vgstar_prime, \
                      'Lg_prime': Lg_prime, 'Lgstar_prime': Lgstar_prime, \
                      'Ve_prime': Ve_prime, 'Vestar_prime': Vestar_prime, \
                      'delta_Le': delta_Le, 'delta_Lestar': delta_Lestar, \
                      'leakage1': leakage1, 'leakage2': leakage2,'leakage3': leakage3, \
                      'leakage3': leakage3, 'chg_extraction': chg_extraction, 'chg_production': chg_production, \
                      'chg_consumption': chg_consumption,'chg_Qeworld':chg_Qeworld, 'pai_g': pai_g, \
                      'subsidy_ratio': subsidy_ratio, 'welfare':welfare, 'welfare_noexternality':welfare_noexternality}))



     