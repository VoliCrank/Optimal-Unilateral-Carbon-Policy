import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.integrate import quad


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

    ## hat values for jxbar, j0bar and jmbar
    jxbar_hat = pe ** (-alpha * theta) * (pe + tb_mat[0]) ** (-(1 - alpha) * theta) / (
            df['jxbar'] * pe ** (-alpha * theta) * (pe + tb_mat[0]) ** (-(1 - alpha) * theta) + (
            1 - df['jxbar']) * (pe + (1 - alpha) * tb_mat[0]) ** (-theta))
    j0_hat = (pe + tb_mat[0]) ** (-(1 - alpha) * theta) / (
            df['jxbar'] * (pe + tb_mat[0]) ** (-(1 - alpha) * theta) + (1 - df['jxbar']) * pe ** (
            -(1 - alpha) * theta))
    jmbar_hat = 1

    if tax_scenario['tax_sce'] == 'Unilateral':
        te = varphi
        tb_mat[1] = 1

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
        jxbar_hat = (1 - df['jxbar']) ** (-1) / (
                (1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0] / pe) ** (theta * (1 - alpha)))
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (
                1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0] / pe) ** (theta * (1 - alpha)))
        tb_mat[1] = 1

    if tax_scenario['tax_sce'] == 'EC_hybrid':
        te = varphi
        jxbar_hat = 1
        jmbar_hat = 1
        tb_mat[1] = 1

    if tax_scenario['tax_sce'] == 'PC_hybrid':
        te = tb_mat[0]
        jxbar_hat = (1 - df['jxbar']) ** (-1) / (
                (1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[1] * tb_mat[0]) / pe) ** (
                theta * (1 - alpha)))
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (
                1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0]) / (pe + tb_mat[0])) ** (
                theta * (1 - alpha)))

    if tax_scenario['tax_sce'] == 'EP_hybrid':
        te = tb_mat[1]
        jxbar_hat = (1 - df['jxbar']) ** (-1) / (
                (1 - df['jxbar']) ** (-1) - 1 + (1 + tb_mat[0] / pe) ** (theta * (1 - alpha)))
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (
                1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * (1 + tb_mat[0] / pe) ** (theta * (1 - alpha)))

    if tax_scenario['tax_sce'] == 'EPC_hybrid':
        te = varphi
        jxbar_hat = (1 - df['jxbar']) ** (-1) / (
                (1 - df['jxbar']) ** (-1) - 1 + (1 + (tb_mat[0] - tb_mat[1] * tb_mat[0]) / pe) ** (
                theta * (1 - alpha)))
        jmbar_hat = (1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1)) / (
                1 + ((1 - df['jmbar']) ** (-1) - 1) ** (-1) * ((pe + tb_mat[0]) / (pe + tb_mat[0])) ** (
                theta * (1 - alpha)))

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
def compute_qe(petbte, epsilonS, epsilonSstar, logit, beta, gamma, pe, df):
    if logit == 1:
        epsilonS = beta * (1 - gamma) / (1 - gamma + gamma * petbte ** beta)
        epsilonSstar = beta * (1 - gamma) / (1 - gamma + gamma * pe ** beta)
        Qe_hat = (petbte) ** beta / (1 - gamma + gamma * (petbte) ** beta)
        Qestar_hat = pe ** beta / (1 - gamma + gamma * pe ** beta)

    ## compute hat values    
    Qe_hat = (petbte) ** epsilonS
    Qestar_hat = pe ** epsilonSstar

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
    
    # Ce^y, home domestic production for domestic consumption
    CeHH_hat = (pe + tb_mat[0]) ** (-epsilonD) * jmbar_hat ** (1 + (1 - sigma) / theta)
    CeHH_prime = df['CeHH'] * CeHH_hat

    # Ce^x, home export
    CeFH1_hat = (pe + tb_mat[0]) ** (-epsilonDstar) * j0_hat ** (1 + (1 - sigmastar) / theta)
    CeFH2_hat = (1 + (1 - sigmastar) / theta) * ((1 - df['jxbar']) / df['jxbar']) ** (sigmastar / theta) * pe ** (
        -epsilonDstar) * (1 + tb_mat[0] / pe) ** (-alpha) * (Bfunvec2_prime - Bfunvec1_prime) / df['jxbar'] ** (
                        1 + (1 - sigmastar) / theta)
    CeFH1_prime = df['CeFH'] * CeFH1_hat
    CeFH2_prime = df['CeFH'] * CeFH2_hat
    CeFH_hat = CeFH1_hat + CeFH2_hat
    
    if tax_scenario['Base'] == 1:
        CeFH_hat = pe ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar) / theta)

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        CeFH_hat = (pe + tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (1 + (1 - sigmastar) / theta)

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        CeFH_hat = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) ** (-epsilonDstar) * jxbar_hat ** (
                1 + (1 - sigmastar) / theta)

    if np.isnan(CeFH_hat) == True:
        CeFH_hat = 0
     
    # final value for Ce^x
    CeFH_prime = df['CeFH'] * CeFH_hat

    # Ce^m, home imports
    CeHF_hat = (pe + tb_mat[0]) ** (-epsilonD)
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        CeHF_hat = (pe) ** (-epsilonD) * ((1 - jmbar_prime) / (1 - df['jmbar'])) ** (1 + (1 - sigma) / theta)

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        CeHF_hat = (pe + tb_mat[0]) ** (-epsilonD) * ((1 - jmbar_prime) / (1 - df['jmbar'])) ** (
                1 + (1 - sigma) / theta)

    # final value for Ce^m
    CeHF_prime = df['CeHF'] * CeHF_hat
    
    # Ce^y*, foreign production for foreign consumption
    CeFF_prime = df['CeFF'] * ((1 - jxbar_prime) / (1 - df['jxbar'])) ** (1 + (1 - sigmastar) / theta) * pe ** (
        -epsilonDstar)

    return CeHH_prime, CeFH1_prime, CeFH2_prime, CeFH_prime, CeHF_prime, CeFF_prime, CeFH_hat, CeFH1_hat, CeHF_hat


## input: pe (price of energy), tb_mat, jvals (import/export threshold values)
##        consvals (tuple of energy consumption values), df, tax_scenario, ParaList
## output: value of goods (import, export, domestic, foreign)
def comp_vg(pe, tb_mat, jvals, consvals, df, tax_scenario, ParaList):
    
    # unpack values from tuples
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    CeHH_prime, CeFH1_prime, CeFH2_prime, CeFH_prime, CeHF_prime, CeFF_prime, CeFH_hat, CeFH1_hat, CeHF_hat, Ce_prime, Cestar_prime = consvals
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList

    # value of home and foreign goods
    VgHH = df['CeHH'] / (1 - alpha)
    VgFF = df['CeFF'] / (1 - alpha)

    # value of home export of goods in baseline
    VgFH = df['CeFH'] / (1 - alpha)
    # VgFH_prime = VgFH * pe ** ((1 - sigmastar) * (1 - alpha)) * (1 - (1 - jxbar_prime) ** (1 + (1 - sigmastar)/theta))/ (df['jxbar'] * (1 - df['jxbar']) ** ( (1-sigmastar)/theta))
    VgFH1_hat = (pe + tb_mat[0]) * CeFH1_hat
    VgFH2_hat = pe ** (1 - epsilonDstar) * ((1 - j0_prime) ** (1 + (1 - sigmastar) / theta) - (1 - jxbar_prime) ** (
            1 + (1 - sigmastar) / theta)) / (df['jxbar'] * (1 - df['jxbar']) ** ((1 - sigmastar) / theta))
    VgFH1_prime = VgFH * VgFH1_hat
    VgFH2_prime = VgFH * VgFH2_hat
    VgFH_hat = VgFH1_hat + VgFH2_hat

    if tax_scenario['Base'] == 1:
        VgFH_hat = pe * CeFH_hat

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        VgFH_hat = (pe + tb_mat[0]) * CeFH_hat

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        VgFH_hat = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * CeFH_hat

    if np.isnan(VgFH_hat) == True:
        VgFH_hat = 0
        
    # final value of home export of good    
    VgFH_prime = VgFH * VgFH_hat

    # value of home import of good in baseline
    VgHF = df['CeHF'] / (1 - alpha)
    VgHF_hat = (pe + alpha * tb_mat[0]) * CeHF_hat

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        VgHF_hat = pe * CeHF_hat

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        VgHF_hat = (pe + alpha * tb_mat[0]) * CeHF_hat
    
    # final value of home import of good
    VgHF_prime = VgHF * VgHF_hat

    return VgHH, VgFF, VgFH1_prime, VgFH2_prime, VgFH_prime, VgHF_prime, VgFH, VgHF


## input: pe (price of energy), tb_mat (border adjustments),
##        consvals: tuple of energy consupmtion values, tax_scenario
## output: Ve_prime, Vestar_prime (final values of home and foreign energy consumption)
def comp_ve(pe, tb_mat, consvals, tax_scenario):
    # unpack parameters
    CeHH_prime, CeFH1_prime, CeFH2_prime, CeFH_prime, CeHF_prime, CeFF_prime, CeFH_hat, CeFH1_hat, CeHF_hat, Ce_prime, Cestar_prime = consvals
    
    Ve_prime = (pe + tb_mat[0]) * Ce_prime
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Ve_prime = (pe + tb_mat[0]) * CeHH_prime + pe * CeHF_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Ve_prime = (pe + tb_mat[0]) * CeHH_prime + (pe + tb_mat[0]) * CeHF_prime
    
    Vestar_prime = (pe + tb_mat[0]) * CeFH1_prime + pe * CeFH2_prime + pe * CeFF_prime

    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Vestar_prime = pe * Cestar_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Vestar_prime = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * CeFH_prime + pe * CeFF_prime

    return Ve_prime, Vestar_prime


## input: pe (price of energy), tb_mat (border adjustments)
##        consvals (tuple of energy consumption values)
##        VgFH_prime (value of home export of goods)
##        ParaList, df, tax_scenario
## output: Vg, Vg_prime, Vgstar, Vgstar_prime (values of home and foreign total spending on goods)
##         non prime values returned to simplify later computations
def comp_vgfin(pe, tb_mat, consvals, VgFH_prime, ParaList, df, tax_scenario):
    
    # unpack parameters
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    CeHH_prime, CeFH1_prime, CeFH2_prime, CeFH_prime, CeHF_prime, CeFF_prime, CeFH_hat, CeFH1_hat, CeHF_hat, Ce_prime, Cestar_prime = consvals

    # initial home spending on goods
    Vg = df['Ce'] / (1 - alpha)
    Vg_prime_hat = (pe + tb_mat[0]) * Ce_prime / df['Ce']
    
    # final home spending on goods
    Vg_prime = Vg * Vg_prime_hat
    
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Vg_prime = CeHH_prime / (1 - alpha) * (pe + tb_mat[0]) + CeHF_prime / (1 - alpha) * pe

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Vg_prime = CeHH_prime / (1 - alpha) * (pe + tb_mat[0]) + CeHF_prime / (1 - alpha) * (pe + tb_mat[0])

    # initial foreign spending on goods
    Vgstar = df['Cestar'] / (1 - alpha)
    
    # final foreign spending on goods
    Vgstar_prime = VgFH_prime + CeFF_prime / (1 - alpha) * pe
    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Vgstar_prime = Cestar_prime / (1 - alpha) * pe

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Vgstar_prime = CeFF_prime / (1 - alpha) * pe + CeFH_prime / (1 - alpha) * (
                pe + tb_mat[0] - tb_mat[1] * tb_mat[0])

    return Vg, Vg_prime, Vgstar, Vgstar_prime


## input: pe (price of energy), tb_mat (border adjustments)
##        Ge_prime/Gestar_prime (energy used in home/foreign production)
##        consvals (tuple of consumptions values), ParaList, df, tax_scenario
## output: Lg_prime/Lgstar_prime (labour employed in production in home and foreign)
def comp_lg(pe, tb_mat, Ge_prime, Gestar_prime, consvals, ParaList, df, tax_scenario):
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    CeHH_prime, CeFH1_prime, CeFH2_prime, CeFH_prime, CeHF_prime, CeFF_prime, CeFH_hat, CeFH1_hat, CeHF_hat, Ce_prime, Cestar_prime = consvals

    # initial value
    Lg = alpha / (1 - alpha) * df['Ge']
    # final value, given current pe and tb_mat
    Lg_prime = alpha / (1 - alpha) * (pe + tb_mat[0]) * Ge_prime
    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Lg_prime = alpha / (1 - alpha) * (pe + tb_mat[0]) * CeHH_prime + alpha / (1 - alpha) * pe * CeFH_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Lg_prime = alpha / (1 - alpha) * (pe + tb_mat[0]) * CeHH_prime + alpha / (1 - alpha) * (
                pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * CeFH_prime

    Lgstar = alpha / (1 - alpha) * df['Gestar']
    Lgstar_prime = alpha / (1 - alpha) * (pe + tb_mat[0]) * CeHF_prime + alpha / (1 - alpha) * pe * CeFF_prime
    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Lgstar_prime = alpha / (1 - alpha) * pe * Gestar_prime
    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Lgstar_prime = alpha / (1 - alpha) * (pe + tb_mat[0]) * CeHF_prime + alpha / (1 - alpha) * pe * CeFF_prime

    return Lg, Lg_prime, Lgstar, Lgstar_prime

## input: Lg, Lg_prime, Lgstar, Lgstar_prime (labour employed in home and foreign production)
##        Qeworld_prime (world energy extraction),
##        Vg, Vgstar (hat value for home and foreign spending on goods), df,
##        jvals (tuple of import/export margins), pe (price of energy),
##        petbte (price faced by home extractor, lower bounded by 0)
##        tb_mat (border adjustments), tax_scenario, varphi, ParaList
## output: compute change in Le/Lestar (labour in home/foreign extraction)
##         change home utility
def comp_delta(Lg, Lg_prime, Lgstar, Lgstar_prime, Qeworld_prime, Vg, Vgstar, df, jvals, pe, petbte, tb_mat,
               tax_scenario, varphi, ParaList):
    
    # unpack parameters
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    
    # change in labour in home/foreign extraction
    delta_Le = (epsilonS / (epsilonS + 1)) * df['Qe'] * (petbte ** (epsilonS + 1) - 1)
    delta_Lestar = (epsilonSstar / (epsilonSstar + 1)) * df['Qestar'] * (pe ** (epsilonSstar + 1) - 1)

    def Func(a, beta, gamma):
        return (((1 - gamma) * a ** beta) / (1 - gamma + gamma * a ** beta) ** 2)

    if logit == 1:
        delta_Le = beta * df['Qe'] * quad(Func, 1, petbte, args=(beta, gamma))[0]
        delta_Lestar = beta * df['Qestar'] * quad(Func, 1, pe, args=(beta, gamma))[0]

    delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
              + Vg * (alpha - 1) * math.log(pe + tb_mat[0]) + Vgstar * (1 / theta) * math.log(
        df['jxbar'] / j0_prime * (pe + tb_mat[0]) ** (-(1 - alpha) * theta)) \
              - varphi * (Qeworld_prime - df['Qeworld'])

    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'purete' or tax_scenario[
        'tax_sce'] == 'EC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
                  + Vg * (alpha - 1) * math.log(pe + tb_mat[0]) + Vgstar * (alpha - 1) * math.log(pe) \
                  - varphi * (Qeworld_prime - df['Qeworld'])

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
                  + Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1 / theta * math.log(df['jmbar'] / jmbar_prime)) \
                  + Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1 / theta * math.log(df['jxbar'] / jxbar_prime)) \
                  - varphi * (Qeworld_prime - df['Qeworld'])

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        delta_U = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) \
                  + Vg * ((alpha - 1) * math.log(pe + tb_mat[0]) + 1 / theta * math.log(df['jmbar'] / jmbar_prime)) \
                  + Vgstar * ((alpha - 1) * math.log(pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) + 1 / theta * math.log(
            df['jxbar'] / jxbar_prime)) \
                  - varphi * (Qeworld_prime - df['Qeworld'])
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


## input: pe (price of energy), tb_mat (border adjustments), CeHH_prime (home consumption of energy on goods produced at home)
##        CeFH_prime (energy in home export), ParaList, tax_scenario, df
## output: marginal leakage (-(partial Gestar / partial re) / (partial Ge / partial re))
##         for different tax scenarios.
def comp_mleak(pe, tb_mat, jvals, CeHH_prime, CeFH_prime, ParaList, tax_scenario, df):
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals

    ## re is different for puretp/EP and PC/EPC
    re = (pe + tb_mat[0]) / pe
    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        re = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) / pe

    tejxj = (1 + (1 - sigmastar) / theta) * df['CeFH'] * jxbar_hat ** ((1 - sigmastar) / theta) * (pe * re) ** (
        -epsilonDstar) / df['jxbar']
    ejyj = (1 + (1 - sigmastar) / theta) * df['CeHH'] * jmbar_hat ** ((1 - sigma) / theta) * (pe * re) ** (-epsilonD) / \
           df['jmbar']
    djxdre = -(theta * (1 - alpha)) * jxbar_prime * (1 - jxbar_prime) / re
    djmdre = -(theta * (1 - alpha)) * jmbar_prime * (1 - jmbar_prime) / re
    
    leaknum = re * (ejyj * djmdre + tejxj * djxdre)
    leakdenum = ejyj * djmdre + tejxj * djxdre - epsilonD * CeHH_prime / re - epsilonDstar * CeFH_prime / re
    leak = leaknum / leakdenum

    leaknum2 = re * tejxj * djxdre
    leakdenum2 = tejxj * djxdre - epsilonDstar * CeFH_prime / re
    leak2 = leaknum2 / leakdenum2

    return leak, leak2


## input: consval (tuple of consumption values), jvals (tuple of import/export margins),
##        Ge_prime/Gestar_prime (home/foreign production energy use),
##        Qe_prime/Qestar_prime/Qeworld_prime (home/foreign/world energy extraction),
##        VgFH2_prime (intermediate value for value of home exports),
##        pe (price of energy), tax_scenario, tb_mat (border adjustments), te (extraction tax)
##        varphi, ParaList, df
## output: objective values
##         diff (difference between total consumption and extraction)
##         diff1 & diff3 (equation to compute wedge and border rebate as in table 4 in paper)
def comp_diff(consvals, jvals, Ge_prime, Gestar_prime, Qe_prime, Qestar_prime, Qeworld_prime, VgFH2_prime, pe,
              tax_scenario, tb_mat, te, varphi, ParaList, df):
    
    # unpack parameters
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    CeHH_prime, CeFH1_prime, CeFH2_prime, CeFH_prime, CeHF_prime, CeFF_prime, CeFH_hat, CeFH1_hat, CeHF_hat, Ce_prime, Cestar_prime = consvals
    alpha, theta, sigma, sigmastar, epsilonD, epsilonDstar, epsilonS, epsilonSstar, beta, gamma, logit = ParaList

    # compute marginal leakage
    leak, leak2 = comp_mleak(pe, tb_mat, jvals, CeHH_prime, CeFH_prime, ParaList, tax_scenario, df)

    # compute world energy consumption and necessary elasticities
    Ceworld_prime = Ce_prime + Cestar_prime
    epsilonSw = (Qe_prime) * epsilonS / Qeworld_prime + Qestar_prime * epsilonSstar / Qeworld_prime
    epsilonDw = Ce_prime * epsilonD / Ceworld_prime + Cestar_prime * epsilonDstar / Ceworld_prime
    epsilonG = CeHH_prime * epsilonD / Ge_prime + CeFH_prime * epsilonDstar / Ge_prime
    epsilonGstar = CeHF_prime * epsilonD / Gestar_prime + CeFF_prime * epsilonDstar / Gestar_prime
    epsilonGw = epsilonDw

    # initialize diff values
    diff1 = 0
    diff2 = 0
    
    if tax_scenario['tax_sce'] == 'Unilateral':
        S = (pe + tb_mat[0]) * CeFH2_prime / (1 - alpha) - VgFH2_prime
        numerator = varphi * epsilonSstar * Qestar_prime - sigmastar * (1 - alpha) * S
        denominator = epsilonSstar * Qestar_prime + epsilonDstar * CeFF_prime
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        
    if tax_scenario['tax_sce'] == 'purete':
        numerator = varphi * epsilonSstar * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + epsilonDw * Ceworld_prime
        # te = varphi - consumption wedge
        diff1 = (varphi - te) * denominator - numerator
        
    if tax_scenario['tax_sce'] == 'puretc':
        numerator = varphi * epsilonSw * Qeworld_prime
        denominator = epsilonSw * Qeworld_prime + epsilonDstar * Cestar_prime
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        
    if tax_scenario['tax_sce'] == 'puretp':
        numerator = varphi * epsilonSw * Qeworld_prime
        denominator = epsilonSw * Qeworld_prime + epsilonGstar * Gestar_prime + leak * epsilonG * Ge_prime
        # border adjustment = (1-leakage) consumption wedge
        diff1 = tb_mat[0] - (1 - leak) * numerator / denominator
        
    if tax_scenario['tax_sce'] == 'EC_hybrid':
        numerator = varphi * epsilonSstar * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + epsilonDstar * Cestar_prime
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        
    if tax_scenario['tax_sce'] == 'PC_hybrid':
        numerator = varphi * epsilonSw * Qeworld_prime
        denominator = epsilonSw * Qeworld_prime + epsilonDstar * CeFF_prime + leak2 * epsilonDstar * CeFH_prime
        diff1 = (tb_mat[0] * denominator - numerator) 
        # border rebate for exports tb[1] * tb[0] = leakage * tc
        diff2 = (tb_mat[1] * tb_mat[0]) * denominator - (leak2) * numerator
        
    if tax_scenario['tax_sce'] == 'EP_hybrid':
        numerator = varphi * epsilonSstar * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + epsilonGstar * Gestar_prime + leak * epsilonG * Ge_prime
        # tp equal to (1-leakge) * consumption wedge
        diff1 = tb_mat[0] * denominator - (1 - leak) * numerator
        # requires nominal extraction tax to be equal to te + tp
        diff2 = (varphi - tb_mat[1]) * denominator - leak * numerator
        
    if tax_scenario['tax_sce'] == 'EPC_hybrid':
        numerator = varphi * epsilonSw * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + epsilonDstar * CeFF_prime + leak2 * epsilonDstar * CeFH_prime
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        # border rebate = leakage * consumption wedge
        diff2 = (tb_mat[0] * tb_mat[1]) * denominator - (leak2) * numerator

    # world extraction = world consumption
    diff = Qe_prime + Qestar_prime - (CeHH_prime + CeFH_prime + CeHF_prime + CeFF_prime)

    return diff, diff1, diff2

# assgin values to return later
def assign_val(Ge_prime, Gestar_prime, Lg_prime, Lgstar_prime, Qe_prime, Qestar_prime, Qeworld_prime, Ve_prime,
               Vestar_prime, VgFH1_prime, VgFH2_prime, VgFH_prime, VgHF_prime, Vg_prime, Vgstar_prime, chg_Qeworld,
               chg_consumption, chg_extraction, chg_production, delta_Le, delta_Lestar, leakage1, leakage2, leakage3,
               pai_g, pe, subsidy_ratio, varphi, welfare, welfare_noexternality, jvals, consvals):
    j0_hat, j0_prime, jxbar_hat, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    CeHH_prime, CeFH1_prime, CeFH2_prime, CeFH_prime, CeHF_prime, CeFF_prime, CeFH_hat, CeFH1_hat, CeHF_hat, Ce_prime, Cestar_prime = consvals
    return (pd.Series({'varphi': varphi, 'pe': pe, 'tb': 0, 'prop': 0, 'te': 0, 'jxbar_prime': jxbar_prime,
                       'jmbar_prime': jmbar_prime, 'j0_prime': j0_prime, 'Qe_prime': Qe_prime,
                       'Qestar_prime': Qestar_prime, 'Qeworld_prime': Qeworld_prime,
                       'CeHH_prime': CeHH_prime, 'CeFH_prime': CeFH_prime, 'CeHF_prime': CeHF_prime,
                       'CeFF_prime': CeFF_prime, 'Ge_prime': Ge_prime, 'Ce_prime': Ce_prime,
                       'Gestar_prime': Gestar_prime,
                       'Cestar_prime': Cestar_prime, 'VgFH_prime': VgFH_prime, 'VgHF_prime': VgHF_prime,
                       'VgFH1_prime': VgFH1_prime, 'VgFH2_prime': VgFH2_prime, 'CeFH1_prime': CeFH1_prime,
                       'CeFH2_prime': CeFH2_prime,
                       'Vg_prime': Vg_prime, 'Vgstar_prime': Vgstar_prime, 'Lg_prime': Lg_prime,
                       'Lgstar_prime': Lgstar_prime,
                       'Ve_prime': Ve_prime, 'Vestar_prime': Vestar_prime, 'delta_Le': delta_Le,
                       'delta_Lestar': delta_Lestar,
                       'leakage1': leakage1, 'leakage2': leakage2, 'leakage3': leakage3,
                       'chg_extraction': chg_extraction, 'chg_production': chg_production,
                       'chg_consumption': chg_consumption, 'chg_Qeworld': chg_Qeworld, 'pai_g': pai_g,
                       'subsidy_ratio': subsidy_ratio, 'welfare': welfare,
                       'welfare_noexternality': welfare_noexternality}))