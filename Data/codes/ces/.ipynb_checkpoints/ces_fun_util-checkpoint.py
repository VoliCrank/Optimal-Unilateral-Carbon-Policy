import math
import pandas as pd
from scipy.integrate import quad
from sympy import *
import sympy
from scipy.integrate import quad
x = symbols('x')

rho = -0.5
alpha = 0.0690161379156130
## define CES production function and its derivative
def g(p, rho = rho, alpha = alpha):
    if rho == 0:
        return alpha**(-alpha) * (1-alpha)**(-(1-alpha)) * p**alpha
    else:
        t1 = (1-alpha)**(1/(1-rho))
        t2 = alpha**(1/(1-rho)) * p**(-rho/(1-rho))
        return (t1 + t2)**(-(1-rho)/rho)

#def gprime(p, rho = rho, alpha = alpha):
#    if rho == 0:
#        return (alpha/(1-alpha))**(1-alpha) * p**(-(1-alpha))
#    else:
#        res = diff(g(x, rho),x).subs(x,p)
#        if type(res) == sympy.core.numbers.Float:
#            res = float(res)
#        return res
    
def gprime(p, rho = rho, alpha = alpha):
    if rho == 0:
        return (alpha/(1-alpha))**(1-alpha) * p**(-(1-alpha))
    else:
        t1 = (1-alpha)**(1/(1-rho))
        t2 = alpha**(1/(1-rho)) * p**(-rho/(1-rho))
        coef = alpha **(1/(1-rho)) * p**(-rho/(1-rho) - 1)
        return (t1 + t2)**(-(1-rho)/rho - 1) * coef

def k(p):
    return gprime(p) / (g(p)-p*gprime(p))

## Dstar(p, sigmastar) corresponds to D* in paper while Dstar(p, sigma) corresponds to D in paper
def Dstar(p, sigmastar):
    return gprime(p) * g(p)**(-sigmastar)

def Dstarprime(p, sigmastar):
    return diff(Dstar(x, sigmastar),x).subs(x,p)

def D1star(p,t,sigmastar):
    return g(p)**(-sigmastar) * gprime(p + t)

def D1starprime(p,t,sigmastar):
    return diff(D1star(x,t,sigmastar),x).subs(x,p)


## input: Paralist (list of parameters selected by hand), pe (price of energy), te (extraction tax), varphi (social cost of carbon)
##        tb_mat (tb_mat[0] = border adjustment,
##                tb_mat[1] = proportion of tax rebate on exports or extraction tax (in the case of EP_hybrid))
## output: te (extraction tax)
##         jxbar_hat, jmbar_hat, j0_hat (hat algebra for import/export threshold, 
##                                       final value obtained by multiplying by df['jxbar'] / df['jmbar'])
##         tb_mat (modify tb_mat[1] value to a default value for cases that do not use tb_mat[1])
def computejbar(ParaList, pe, te, varphi, tb_mat, tax_scenario, df):
    
    # unpack parameters
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    
    ## new formulation
    Cex = df['CeFH']
    Ceystar = df['CeFF']
    
    jxbar_hat =  g(pe + tb_mat[0])**(-theta) * Cex / (g(pe+tb_mat[0])**(-theta) * Cex + (g(pe) + tb_mat[0] * gprime(pe))**(-theta) * Ceystar) / df['jxbar']
    j0_hat = g(pe + tb_mat[0])**(-theta) * Cex / (g(pe+tb_mat[0])**(-theta) * Cex + (g(pe))**(-theta) * Ceystar) / df['jxbar'] 
    jmbar_hat = 1
    
    
    if tax_scenario['tax_sce'] == 'Unilateral':
        te = varphi
        tb_mat[1] = 1

    if tax_scenario['tax_sce'] == 'global':
        te = varphi
        tb_mat[0] = 0
        jxbar_hat = 1
        jmbar_hat = 1

    if tax_scenario['tax_sce'] == 'purete':
        jxbar_hat = 1
        jmbar_hat = 1

    if tax_scenario['tax_sce'] == 'puretc':
        te = tb_mat[0]
        jxbar_hat = 1
        jmbar_hat = 1
        tb_mat[1] = 1

    if tax_scenario['tax_sce'] == 'puretp':
        te = tb_mat[0]
        ve = pe+tb_mat[0]
        jmbar_hat = df['CeHH'] * (g(pe) / g(ve))**theta / (df['CeHH'] * (g(pe) / g(ve))**theta + df['CeHF']) / df['jmbar']
        jxbar_hat = df['CeFH'] * (g(pe) / g(ve))**theta / (df['CeFH'] * (g(pe) / g(ve))**theta + df['CeFF']) / df['jxbar']
        tb_mat[1] = 1

    if tax_scenario['tax_sce'] == 'EC_hybrid':
        te = varphi
        jxbar_hat = 1
        jmbar_hat = 1
        tb_mat[1] = 1

    if tax_scenario['tax_sce'] == 'PC_hybrid':
        te = tb_mat[0]
        ve = pe + tb_mat[0] - tb_mat[0] * tb_mat[1]
        jmbar_hat = 1
        jxbar_hat = df['CeFH'] * (g(pe) / g(ve))**theta / (df['CeFH'] * (g(pe) / g(ve))**theta + df['CeFF']) / df['jxbar']

    if tax_scenario['tax_sce'] == 'EP_hybrid':
        te = tb_mat[1]
        ve = pe+tb_mat[0]
        jmbar_hat = df['CeHH'] * (g(pe) / g(ve))**theta / (df['CeHH'] * (g(pe) / g(ve))**theta + df['CeHF']) / df['jmbar']
        jxbar_hat = df['CeFH'] * (g(pe) / g(ve))**theta / (df['CeFH'] * (g(pe) / g(ve))**theta + df['CeFF']) / df['jxbar']

    if tax_scenario['tax_sce'] == 'EPC_hybrid':
        te = varphi
        ve = pe + tb_mat[0] - tb_mat[0] * tb_mat[1]
        jmbar_hat = 1
        jxbar_hat = df['CeFH'] * (g(pe) / g(ve))**theta / (df['CeFH'] * (g(pe) / g(ve))**theta + df['CeFF']) / df['jxbar']
        


    return te, jxbar_hat, jmbar_hat, j0_hat, tb_mat


## input: j0_prime, jxbar_prime, theta and sigmastar
## output: compute values for the incomplete beta functions, from 0 to j0_prime and from 0 to jxbar_prime
def imcomp_betas(j0_prime, jxbar_prime, theta, sigmastar):
    def tempFunction(i, theta, sigmastar):
        return (i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1))

    Bfunvec1_prime = quad(tempFunction, 0, j0_prime, args=(theta, sigmastar))[0]
    Bfunvec2_prime = quad(tempFunction, 0, jxbar_prime, args=(theta, sigmastar))[0]

    return (Bfunvec1_prime, Bfunvec2_prime)


## input: petbte (pe + tb - te, price faced by home extractors. Becomes 0 if pe + tb - te < 0)
##        epsilonS, epsilonSstar, logit, beta, gamma (parameters set by user)
##        pe (price of energy)
## output: home and foreign extraction values
def compute_qe(tax_scenario, petbte, epsilonS, epsilonSstar, logit, beta, gamma, pe, df):
    if logit == 1:
        epsilonS = beta * (1 - gamma) / (1 - gamma + gamma * petbte ** beta)
        epsilonSstar = beta * (1 - gamma) / (1 - gamma + gamma * pe ** beta)
        Qe_hat = (petbte) ** beta / (1 - gamma + gamma * (petbte) ** beta)
        Qestar_hat = pe ** beta / (1 - gamma + gamma * pe ** beta)

    ## compute hat values    
    Qe_hat = (petbte) ** epsilonS
    Qestar_hat = pe ** epsilonSstar
    if tax_scenario['tax_sce'] == 'global':
        Qestar_hat = petbte ** epsilonSstar

    ## compute final values
    Qe_prime = df['Qe'] * Qe_hat
    Qestar_prime = df['Qestar'] * Qestar_hat

    return Qe_prime, Qestar_prime


## input: pe (price of energy), tb_mat (border adjustment and export rebate/extraction tax, depending on tax scenario)
##        jvals(tuple of jxbar, jmbar, j0 and their hat values (to simplify later computation))
##        ParaList (tuple of user selected parameter)
##        df, tax_scenario
## output: detailed energy consumption values (home, import, export, foreign 
##         and some hat values for simplifying calculation in later steps)
def comp_ce(pe, tb_mat, jvals, ParaList, df, tax_scenario):
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    
    # compute incomplete beta values
    Bfunvec1_prime, Bfunvec2_prime = imcomp_betas(j0_prime, jxbar_prime, theta, sigmastar)
    
    #### new Cey, jmbar_hat = 1 if not pure tp or EP hybrid
    Cey_prime = Dstar(pe+tb_mat[0], sigma)/ (Dstar(1,sigma)) * df['CeHH'] * (jmbar_hat) **(1+(1-sigma)/theta)

    #### new Cex1, Cex2
    Cex1_hat = Dstar(pe+tb_mat[0], sigmastar) / Dstar(1,sigmastar) * (j0_hat) ** (1+ (1-sigmastar)/theta) 
    
    const = g(pe)**(-sigmastar) * gprime(pe + tb_mat[0]) / (g(1)**(-sigmastar) * gprime(1))
    frac = ((1-df['jxbar'])/df['jxbar'])**(sigmastar/theta) * (theta + 1 - sigmastar)/theta
    jterm = 1/ df['jxbar']**(1+(1-sigmastar)/theta)
    Cex2_hat = const * frac * jterm * (Bfunvec2_prime- Bfunvec1_prime) 
    
    Cex1_prime = df['CeFH'] * Cex1_hat
    Cex2_prime = df['CeFH'] * Cex2_hat
    Cex_hat = Cex1_hat + Cex2_hat
    
    if tax_scenario['Base'] == 1:
        Cex_hat = Dstar(pe, sigmastar) / Dstar(1,sigmastar) * (jxbar_hat) ** (1+ (1-sigmastar)/theta) 

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        ve = pe + tb_mat[0] 
        Cex_hat = Dstar(ve, sigmastar) / Dstar(1,sigmastar) * jxbar_hat ** (1 + (1 - sigmastar) / theta)

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0] 
        Cex_hat = Dstar(ve, sigmastar) / Dstar(1,sigmastar) * jxbar_hat ** (1 + (1 - sigmastar) / theta)

    # final value for Ce^x
    Cex_prime = df['CeFH'] * Cex_hat

    # Ce^m, home imports
    Cem_hat = Dstar(pe + tb_mat[0], sigma) / Dstar(1,sigma)
    
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Cem_hat = Dstar(pe, sigma) / Dstar(1,sigma) * ((1 - jmbar_prime) / (1 - df['jmbar'])) ** (1 + (1 - sigma) / theta)

    # final value for Ce^m
    Cem_prime = df['CeHF'] * Cem_hat
    
    # Ce^y*, foreign production for foreign consumption
    Ceystar_prime = Dstar(pe, sigmastar) / Dstar(1,sigmastar) * df['CeFF'] * ((1-jxbar_prime)/ (1-df['jxbar']))**(1+(1-sigmastar)/theta)

    return Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Cex_hat, Cex1_hat, Cem_hat




## input: pe (price of energy), tb_mat, jvals (import/export threshold values)
##        consvals (tuple of energy consumption values), df, tax_scenario, ParaList
## output: value of goods (import, export, domestic, foreign)
def comp_vg(pe, tb_mat, jvals, consvals, df, tax_scenario, ParaList):
    
    # unpack values from tuples
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Cex_hat, Cex1_hat, Cem_hat, Ce_prime, Cestar_prime = consvals
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList

    ## BAU values
    scale = g(1) / gprime(1)
    # value of home and foreign goods
    Vgy = df['CeHH'] * scale
    Vgystar = df['CeFF'] * scale
    # value of home export of goods in baseline
    Vgx = df['CeFH'] * scale
    # value of home import of good in baseline
    Vgm = df['CeHF'] * scale
    
    
    ## Value of exports for unilateral optimal
    Vgx1_prime = (g(pe+tb_mat[0]) / g(1))**(1-sigmastar) * j0_hat**(1+(1-sigmastar)/theta) * Vgx
    
    pterm = (g(pe) / g(1))**(1-sigmastar) * Vgx
    num = (1-j0_prime)**((theta + 1 - sigmastar)/theta) - (1-jxbar_prime)**((theta + 1 - sigmastar) / theta)
    denum = df['jxbar'] * (1-df['jxbar'])**((1-sigmastar)/theta)
    Vgx2_prime = pterm * num / denum
    Vgx_hat = (Vgx1_prime + Vgx2_prime)/ Vgx

    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'purete' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Vgx_hat = (g(pe) / g(1)) ** (1-sigmastar)

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        ve = pe + tb_mat[0]
        Vgx_hat = (g(ve) / g(1)) ** (1-sigmastar) * (jxbar_hat) ** (1+(1-sigmastar)/theta)

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
        Vgx_hat = (g(ve) / g(1)) ** (1-sigmastar) * (jxbar_hat) ** (1+(1-sigmastar)/theta)
        
    # final value of home export of good    
    Vgx_prime = Vgx * Vgx_hat

    # value of home import of good
    Vgm_hat = (g(pe+tb_mat[0])/g(1))**(1-sigma)
        
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Vgm_hat = (g(pe) / g(1))**(1-sigma) * ((1-jmbar_prime) / (1-df['jmbar'])) ** (1+(1-sigma)/theta)

    # final value of home import of good
    Vgm_prime = Vgm * Vgm_hat
    
    #Vgystar_prime = (g(pe) / g(1))**(1-sigmastar) * Vgystar    # equivalent to Ceystar_prime * g(pe) / gprime(pe)

    return Vgy, Vgystar, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgm_prime, Vgx, Vgm


## input: pe (price of energy), tb_mat (border adjustments),
##        consvals: tuple of energy consupmtion values, tax_scenario
## output: Ve_prime, Vestar_prime (final values of home and foreign energy consumption)
def comp_ve(pe, tb_mat, consvals, tax_scenario):
    # unpack parameters
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Cex_hat, Cex1_hat, Cem_hat, Ce_prime, Cestar_prime = consvals
    
    Ve_prime = (pe + tb_mat[0]) * Ce_prime
    
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Ve_prime = (pe + tb_mat[0]) * Cey_prime + pe * Cem_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Ve_prime = (pe + tb_mat[0]) * Cey_prime + (pe + tb_mat[0]) * Cem_prime
        
    
    Vestar_prime = (pe + tb_mat[0]) * Cex_prime + pe * Ceystar_prime
    
    if tax_scenario['tax_sce'] == 'Unilateral':
        Vestar_prime = (pe + tb_mat[0]) * Cex1_prime + pe * Cex2_prime + pe * Ceystar_prime

    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Vestar_prime = pe * Cestar_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Vestar_prime = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex_prime + pe * Ceystar_prime

    return Ve_prime, Vestar_prime


## input: pe (price of energy), tb_mat (border adjustments)
##        consvals (tuple of energy consumption values)
##        Vgx_prime (value of home export of goods)
##        ParaList, df, tax_scenario
## output: Vg, Vg_prime, Vgstar, Vgstar_prime (values of home and foreign total spending on goods)
##         non prime values returned to simplify later computations
def comp_vgfin(pe, tb_mat, consvals, vgvals, jvals, ParaList, df, tax_scenario):
    
    # unpack parameters
    Vgx_prime, Vgm_prime = vgvals
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Cex_hat, Cex1_hat, Cem_hat, Ce_prime, Cestar_prime = consvals


    # home spending on goods
    Vg = df['Ce'] * g(1) / gprime(1)
    Vg_prime = (g(pe+tb_mat[0]) / g(1)) **(1-sigma) * Vg
    
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        scale = g(1) / gprime(1)
        # value of home and foreign goods
        Vgy = df['CeHH'] * scale
        Vgy_prime = (g(pe+tb_mat[0]) / g(1))**(1-sigma) * jmbar_hat ** (1+(1-sigma)/theta) * Vgy
        Vg_prime = Vgy_prime + Vgm_prime

    # foreign spending on goods
    Vgstar = df['Cestar'] * g(1) / gprime(1)
    Vgstar_prime = Vgx_prime + Ceystar_prime * g(pe) / gprime(pe)

    return Vg, Vg_prime, Vgstar, Vgstar_prime


## input: pe (price of energy), tb_mat (border adjustments)
##        Ge_prime/Gestar_prime (energy used in home/foreign production)
##        consvals (tuple of consumptions values), ParaList, df, tax_scenario
## output: Lg_prime/Lgstar_prime (labour employed in production in home and foreign)
def comp_lg(pe, tb_mat, Ge_prime, Gestar_prime, consvals, ParaList, df, tax_scenario):
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Cex_hat, Cex1_hat, Cem_hat, Ce_prime, Cestar_prime = consvals

    # labour employed in production in home
    Lg = 1/k(1) * df['Ge']
    Lg_prime = 1/k(pe+tb_mat[0]) * Ge_prime
    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Lg_prime = 1/k(pe+tb_mat[0]) * Cey_prime + 1/k(pe) * Cex_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Lg_prime = 1/k(pe+tb_mat[0]) * Cey_prime + 1/k(pe+tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex_prime

    ## labour employed in foreign production
    Lgstar = 1/k(1) * df['Gestar']
    Lgstar_prime = 1/k(pe+tb_mat[0]) * Cem_prime + 1/k(pe) * Ceystar_prime
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Lgstar_prime = 1/k(pe) * Gestar_prime

    return Lg, Lg_prime, Lgstar, Lgstar_prime

## input: Lg, Lg_prime, Lgstar, Lgstar_prime (labour employed in home and foreign production)
##        Qeworld_prime (world energy extraction),
##        Vg, Vgstar (hat value for home and foreign spending on goods), df,
##        jvals (tuple of import/export margins), pe (price of energy),
##        petbte (price faced by home extractor, lower bounded by 0)
##        tb_mat (border adjustments), tax_scenario, varphi, ParaList
## output: compute change in Le/Lestar (labour in home/foreign extraction)
##         change home utility
def comp_delta(lgvals, vgvals, Qeworld_prime, df, jvals, pe, petbte, tb_mat, tax_scenario, varphi, ParaList):
    
    Lg, Lgstar, Lg_prime, Lgstar_prime = lgvals
    Vg, Vgstar, Vg_prime, Vgstar_prime = vgvals
    
    # unpack parameters
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    
    # change in labour in home/foreign extraction
    delta_Le = (epsilonS / (epsilonS + 1)) * df['Qe'] * ((petbte) ** (epsilonS + 1) - 1)
    delta_Lestar = (epsilonSstar / (epsilonSstar + 1)) * df['Qestar'] * (pe ** (epsilonSstar + 1) - 1)
    if tax_scenario['tax_sce'] == 'global':
        delta_Lestar = (epsilonSstar / (epsilonSstar + 1)) * df['Qestar'] * (petbte ** (epsilonSstar + 1) - 1)

    def Func(a, beta, gamma):
        return (((1 - gamma) * a ** beta) / (1 - gamma + gamma * a ** beta) ** 2)

    if logit == 1:
        delta_Le = beta * df['Qe'] * quad(Func, 1, petbte, args=(beta, gamma))[0]
        delta_Lestar = beta * df['Qestar'] * quad(Func, 1, pe, args=(beta, gamma))[0]
    
    # term that is common across all delta_U calculations
    const = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) - varphi * (Qeworld_prime - df['Qeworld'])
    
    if sigma != 1 and sigmastar != 1:
        delta_U = const + sigma/(sigma - 1) * (Vg_prime - Vg) + sigmastar / (sigmastar-1) * (Vgstar_prime - Vgstar)
        return delta_Le, delta_Lestar, delta_U

    #delta_U = Vg * (alpha - 1) * math.log(pe + tb_mat[0]) + Vgstar * (1 / theta) * math.log(df['jxbar'] / j0_prime * (pe + tb_mat[0]) ** (-(1 - alpha) * theta)) + const
    ## in the unilateral optimal case
    if tax_scenario['tax_sce'] == 'Unilateral':
        
        delta_Vg = -math.log((g(pe+tb_mat[0]) /g(1))) * Vg
        
        delta_Vgstar_t1 = -(1-j0_prime) * math.log(g(pe) / g(1) * ((1-j0_prime) / (1-df['jxbar'])) ** (1/theta)) 
        delta_Vgstar_t2 = -j0_prime * math.log(g(pe+tb_mat[0]) / g(1) * (j0_prime / df['jxbar']) ** (1/theta))
        delta_Vgstar = (delta_Vgstar_t1 + delta_Vgstar_t2) * Vgstar
                                              
    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'purete' or tax_scenario['tax_sce'] == 'EC_hybrid':
        #delta_U = const + Vg * (alpha - 1) * math.log(pe + tb_mat[0]) + Vgstar * (alpha - 1) * math.log(pe) 
        delta_Vg = -math.log(g(pe+tb_mat[0]) /g(1)) * Vg
        delta_Vgstar = -math.log(g(pe) / g(1)) * Vgstar

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        delta_U = const + Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1 / theta * math.log(df['jmbar'] / jmbar_prime)) \
                + Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1 / theta * math.log(df['jxbar'] / jxbar_prime))
        
        #delta_Vg = -math.log(g(pe+tb_mat[0]) / g(1)) * Vg
        ve = pe + tb_mat[0]
        delta_Vg_t1 = -jmbar_prime * math.log(g(ve) / g(1) * (jmbar_prime / df['jmbar']) ** (1/theta))
        delta_Vg_t2 = -(1-jmbar_prime) * math.log(g(pe) / g(1) * ((1-jmbar_prime) / (1-df['jmbar']))**(1/theta))
        delta_Vg = (delta_Vg_t1 + delta_Vg_t2)*Vg
        
        delta_Vgstar_t1 = -jxbar_prime * math.log(g(ve) / g(1) * (jxbar_prime / df['jxbar']) ** (1/theta))
        delta_Vgstar_t2 = -(1-jxbar_prime) * math.log(g(pe) / g(1) * ((1-jxbar_prime) / (1-df['jxbar']))**(1/theta))
        delta_Vgstar = (delta_Vgstar_t1 + delta_Vgstar_t2) * Vgstar

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        #delta_U = const + Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1 / theta * math.log(df['jmbar'] / jmbar_prime)) \
        #        + Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) + 1 / theta * math.log(df['jxbar'] / jxbar_prime))
        ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
        delta_Vg = -math.log(g(pe+tb_mat[0]) / g(1)) * Vg
        delta_Vgstar_t1 = -jxbar_prime * math.log(g(ve) / g(1) * (jxbar_prime / df['jxbar']) ** (1/theta))
        delta_Vgstar_t2 = -(1-jxbar_prime) * math.log(g(pe) / g(1) * ((1-jxbar_prime) / (1-df['jxbar']))**(1/theta))
        delta_Vgstar = (delta_Vgstar_t1 + delta_Vgstar_t2) * Vgstar

        
    delta_U = delta_Vg + delta_Vgstar + const
    return delta_Le, delta_Lestar, delta_U


## input: Qestar_prime (foregin extraction), Gestar_prime (foreign energy use in production)
##        Cestar_prime (foregin energy consumption), Qeworld_prime (world extraction), df
## output: returns average leakage for extraction, production and consumption
def comp_leak(Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime, df):
    leakage1 = -(Qestar_prime - df['Qestar']) / (Qeworld_prime - df['Qeworld'])
    leakage2 = -(Gestar_prime - df['Gestar']) / (Qeworld_prime - df['Qeworld'])
    leakage3 = -(Cestar_prime - df['Cestar']) / (Qeworld_prime - df['Qeworld'])

    return leakage1, leakage2, leakage3


## input: df, Qestar_prime (foreign extraction), Gestar_prime (foreign energy use in production)
##        Cestar_prime (foregin energy consumption), Qeworld_prime (world extraction)
## output: compute change in extraction, production and consumption of energy relative to baseline.
def comp_chg(df, Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime):
    chg_extraction = Qestar_prime - df['Qestar']
    chg_production = Gestar_prime - df['Gestar']
    chg_consumption = Cestar_prime - df['Cestar']
    chg_Qeworld = Qeworld_prime - df['Qeworld']

    return chg_extraction, chg_production, chg_consumption, chg_Qeworld


## input: pe (price of energy), tb_mat (border adjustments), Cey_prime (home consumption of energy on goods produced at home)
##        Cex_prime (energy in home export), ParaList, tax_scenario, df
## output: marginal leakage (-(partial Gestar / partial re) / (partial Ge / partial re))
##         for different tax scenarios.
def comp_mleak(pe, tb_mat, jvals, Cey_prime,Cem_prime, Cex_prime, Ceystar_prime, ParaList, tax_scenario, df):
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals

    ## re is different for puretp/EP and PC/EPC
    ve = (pe + tb_mat[0])
    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        ve = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0])
    
    ## new leakage for PC, EPC
    djxdve = -jxbar_prime * (1-jxbar_prime) * gprime(ve) / g(ve) * theta
    djmdve = -jmbar_prime * (1-jmbar_prime) * gprime(ve) / g(ve) * theta
    
    dceydve = Dstarprime(ve, sigma) / Dstar(ve, sigma) * Cey_prime + (1+(1-sigma)/theta) * djmdve / jmbar_prime * Cey_prime
    dcemdve = 1/(1-jmbar_prime) * (1+(1-sigma)/theta) * (-djmdve) * Cem_prime 
    dcexdve = Dstarprime(ve, sigmastar) / Dstar(ve, sigmastar) * Cex_prime +  (1+(1-sigmastar) / theta) * Cex_prime / jxbar_prime * djxdve
    dceystardve = (1+(1-sigmastar) /theta) * Ceystar_prime * (-djxdve) / (1-jxbar_prime)
    
    leak = -(dceystardve + dcemdve) / (dcexdve + dceydve)
    leak2 = -dceystardve / dcexdve

    return leak, leak2


## input: consval (tuple of consumption values), jvals (tuple of import/export margins),
##        Ge_prime/Gestar_prime (home/foreign production energy use),
##        Qe_prime/Qestar_prime/Qeworld_prime (home/foreign/world energy extraction),
##        Vgx2_prime (intermediate value for value of home exports),
##        pe (price of energy), tax_scenario, tb_mat (border adjustments), te (extraction tax)
##        varphi, ParaList, df
## output: objective values
##         diff (difference between total consumption and extraction)
##         diff1 & diff3 (equation to compute wedge and border rebate as in table 4 in paper)
def comp_diff(consvals, jvals, Ge_prime, Gestar_prime, Qe_prime, Qestar_prime, Qeworld_prime, Vgx2_prime, pe,
              tax_scenario, tb_mat, te, varphi, ParaList, df):
    
    # unpack parameters
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Cex_hat, Cex1_hat, Cem_hat, Ce_prime, Cestar_prime = consvals
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList

    # compute marginal leakage
    leak, leak2 = comp_mleak(pe, tb_mat, jvals, Cey_prime,Cem_prime, Cex_prime, Ceystar_prime, ParaList, tax_scenario, df)

    # compute world energy consumption and necessary elasticities
    Ceworld_prime = Ce_prime + Cestar_prime
    epsilonSw = (Qe_prime) * epsilonS / Qeworld_prime + Qestar_prime * epsilonSstar / Qeworld_prime

    # initialize diff values
    diff1 = 0
    diff2 = 0
    
    if tax_scenario['tax_sce'] == 'Unilateral':
        S = g(pe+tb_mat[0]) / gprime(pe+tb_mat[0]) * Cex2_prime - Vgx2_prime
        num = varphi * epsilonSstar * Qestar_prime -sigmastar * gprime(pe) * S / g(pe)
        denum =epsilonSstar*Qestar_prime + abs(Dstarprime(pe,sigmastar) / Dstar(pe,sigmastar)) * pe * Ceystar_prime
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denum - num
        
    if tax_scenario['tax_sce'] == 'global':
        diff1 = Qeworld_prime - Ceworld_prime
        
        
    if tax_scenario['tax_sce'] == 'purete':
        numerator = varphi * epsilonSstar * Qestar_prime
        dcewdpe = abs(Dstarprime(pe,sigma) / Dstar(pe,sigma) * Cey_prime 
                      + Dstarprime(pe,sigmastar) / Dstar(pe,sigmastar) * Cex_prime 
                      + Dstarprime(pe,sigmastar) / Dstar(pe,sigmastar) * (Ceystar_prime + Cem_prime))
        denominator = epsilonSstar * Qestar_prime + dcewdpe * pe
        
        # te = varphi - consumption wedge
        diff1 = (varphi - te) * denominator - numerator
        
    if tax_scenario['tax_sce'] == 'puretc':
        dcestardpe = abs(Dstarprime(pe,sigmastar) / Dstar(pe,sigmastar) * Cex_prime 
                      + Dstarprime(pe,sigmastar) / Dstar(pe,sigmastar) * (Ceystar_prime))
        
        numerator = varphi * (epsilonS * Qe_prime + epsilonSstar * Qestar_prime)
        denominator = (epsilonS * Qe_prime + epsilonSstar * Qestar_prime) + dcestardpe * pe
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        
    if tax_scenario['tax_sce'] == 'puretp':
        numerator = varphi * epsilonSw * Qeworld_prime
        ## energy price faced by home producers
        ve = pe+tb_mat[0]
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1-jxbar_prime)
        djmbardpe = theta * gprime(pe) / g(pe) * jmbar_prime * (1-jmbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1+(1-sigmastar) / theta) / (1-jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1+(1-sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime
        dcemdpe = abs(Dstarprime(pe,sigma) / Dstar(pe,sigma) - (1+(1-sigma)/theta) / (1-jmbar_prime) * djmbardpe) * Cem_prime
        dceydpe = abs((1+(1-sigma)/theta) / jmbar_prime * djmbardpe) * Cey_prime
                      
        denominator = epsilonSw * Qeworld_prime + (dceystardpe + dcemdpe)* pe - leak * (dcexdpe + dceydpe) *pe
        # border adjustment = (1-leakage) consumption wedge
        diff1 = tb_mat[0] * denominator - (1 - leak) * numerator
        
    if tax_scenario['tax_sce'] == 'EC_hybrid':
        dcestardpe = abs(Dstarprime(pe,sigmastar) / Dstar(pe,sigmastar) * Cex_prime 
                      + Dstarprime(pe,sigmastar) / Dstar(pe,sigmastar) * (Ceystar_prime))
        
        numerator = varphi * epsilonSstar * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + dcestardpe * pe
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        
    if tax_scenario['tax_sce'] == 'PC_hybrid':
        ve = pe+tb_mat[0] - tb_mat[0] * tb_mat[1]
        
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1-jxbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1+(1-sigmastar) / theta) / (1-jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1+(1-sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime
        
        numerator = varphi * epsilonSw * Qeworld_prime
        denominator = epsilonSw * Qeworld_prime + dceystardpe* pe - leak2 * dcexdpe *pe
        
        diff1 = (tb_mat[0] * denominator - numerator) 
        # border rebate for exports tb[1] * tb[0] = leakage * tc
        diff2 = (tb_mat[1] * tb_mat[0]) * denominator - (leak2) * numerator
        
    if tax_scenario['tax_sce'] == 'EP_hybrid':
        ## energy price faced by home producers
        ve = pe+tb_mat[0]
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1-jxbar_prime)
        djmbardpe = theta * gprime(pe) / g(pe) * jmbar_prime * (1-jmbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1+(1-sigmastar) / theta) / (1-jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1+(1-sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime
        dcemdpe = abs(Dstarprime(pe,sigma) / Dstar(pe,sigma) - (1+(1-sigma)/theta) / (1-jmbar_prime) * djmbardpe) * Cem_prime
        dceydpe = abs((1+(1-sigma)/theta) / jmbar_prime * djmbardpe) * Cey_prime
        
        numerator = varphi * epsilonSstar * Qestar_prime 
        denominator = epsilonSstar * Qestar_prime + (dceystardpe + dcemdpe)* pe- leak * (dcexdpe + dceydpe) *pe
        
        # tp equal to (1-leakge) * consumption wedge
        diff1 = tb_mat[0] * denominator - (1 - leak) * numerator
        # requires nominal extraction tax to be equal to te + tp
        diff2 = (varphi - tb_mat[1]) * denominator - leak * numerator
        
    if tax_scenario['tax_sce'] == 'EPC_hybrid':
        ## energy price faced by home producers
        ve = pe+tb_mat[0] - tb_mat[0] * tb_mat[1]
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1-jxbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1+(1-sigmastar) / theta) / (1-jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1+(1-sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime
        
        numerator = varphi * epsilonSw * Qestar_prime
        denominator = epsilonSw * Qestar_prime + dceystardpe* pe- leak2 * dcexdpe *pe
        
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        # border rebate = leakage * consumption wedge
        diff2 = (tb_mat[0] * tb_mat[1]) * denominator - (leak2) * numerator

    # world extraction = world consumption
    diff = Qe_prime + Qestar_prime - (Cey_prime + Cex_prime + Cem_prime + Ceystar_prime)
    #print(diff, diff1)
    return diff, diff1 * 2, diff2 * 2

# assgin values to return later
def assign_val(Ge_prime, Gestar_prime, Lg_prime, Lgstar_prime, Qe_prime, Qestar_prime, Qeworld_prime, Ve_prime,
               Vestar_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgm_prime, Vg_prime, Vgstar_prime, chg_Qeworld,
               chg_consumption, chg_extraction, chg_production, delta_Le, delta_Lestar, leakage1, leakage2, leakage3,
               pai_g, pe, subsidy_ratio, varphi, welfare, welfare_noexternality, jvals, consvals):
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Cex_hat, Cex1_hat, Cem_hat, Ce_prime, Cestar_prime = consvals
    return (pd.Series({'varphi': varphi, 'pe': pe, 'tb': 0, 'prop': 0, 'te': 0, 'jxbar_prime': jxbar_prime,
                       'jmbar_prime': jmbar_prime, 'j0_prime': j0_prime, 'Qe_prime': Qe_prime,
                       'Qestar_prime': Qestar_prime, 'Qeworld_prime': Qeworld_prime,
                       'Cey_prime': Cey_prime, 'Cex_prime': Cex_prime, 'Cem_prime': Cem_prime,
                       'Ceystar_prime': Ceystar_prime, 'Ge_prime': Ge_prime, 'Ce_prime': Ce_prime,
                       'Gestar_prime': Gestar_prime,
                       'Cestar_prime': Cestar_prime, 'Vgx_prime': Vgx_prime, 'Vgm_prime': Vgm_prime,
                       'Vgx1_prime': Vgx1_prime, 'Vgx2_prime': Vgx2_prime, 'Cex1_prime': Cex1_prime,
                       'Cex2_prime': Cex2_prime,
                       'Vg_prime': Vg_prime, 'Vgstar_prime': Vgstar_prime, 'Lg_prime': Lg_prime,
                       'Lgstar_prime': Lgstar_prime,
                       'Ve_prime': Ve_prime, 'Vestar_prime': Vestar_prime, 'delta_Le': delta_Le,
                       'delta_Lestar': delta_Lestar,
                       'leakage1': leakage1, 'leakage2': leakage2, 'leakage3': leakage3,
                       'chg_extraction': chg_extraction, 'chg_production': chg_production,
                       'chg_consumption': chg_consumption, 'chg_Qeworld': chg_Qeworld, 'pai_g': pai_g,
                       'subsidy_ratio': subsidy_ratio, 'welfare': welfare,
                       'welfare_noexternality': welfare_noexternality}))