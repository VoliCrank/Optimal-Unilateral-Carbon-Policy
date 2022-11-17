import math
import numpy as np
import pandas as pd
from sympy import *
from scipy.integrate import quad
from scipy.optimize import minimize

x = symbols('x')

rho = 0
alpha = 0.15

## define CES production function and its derivative
def g(p, rho=rho, alpha=alpha):
    if rho == 0:
        return alpha ** (-alpha) * (1 - alpha) ** (-(1 - alpha)) * p ** alpha
    else:
        t1 = (1 - alpha) ** (1 / (1 - rho))
        t2 = alpha ** (1 / (1 - rho)) * p ** (-rho / (1 - rho))
        return (t1 + t2) ** (-(1 - rho) / rho)

def gprime(p, rho=rho, alpha=alpha):
    if rho == 0:
        return (alpha / (1 - alpha)) ** (1 - alpha) * p ** (-(1 - alpha))
    else:
        t1 = (1 - alpha) ** (1 / (1 - rho))
        t2 = alpha ** (1 / (1 - rho)) * p ** (-rho / (1 - rho))
        coef = alpha ** (1 / (1 - rho)) * p ** (-rho / (1 - rho) - 1)
        return (t1 + t2) ** (-(1 - rho) / rho - 1) * coef

def k(p):
    return gprime(p) / (g(p) - p * gprime(p))

## Dstar(p, sigmastar) corresponds to D*(p) in paper while Dstar(p, sigma) corresponds to D(p) in paper
def Dstar(p, sigmastar):
    return gprime(p) * g(p) ** (-sigmastar)

def Dstarprime(p, sigmastar):
    return diff(Dstar(x, sigmastar), x).subs(x, p)


## class object for finding equilibrium taxes
class tax_eq:
    def __init__(self, pe, te, tb_mat, df, tax_scenario, varphi, paralist):
        self.df = df
        self.tax_scenario = tax_scenario
        self.varphi = varphi
        self.paralist = paralist
        self.prop = 1
        self.te = te
        self.pe = pe
        self.tb_mat = tb_mat
        self.conv = 1
        self.region = df['region_scenario']

    def min_obj(self, props, tbs, pes, bounds=[(0.01, np.inf), (0, np.inf), (0, 1)]):
        for prop in props:
            for tb in tbs:
                for pe in pes:
                    res = minimize(self.find_eq, [pe, tb, prop], bounds=bounds,
                                   method='nelder-mead', tol=0.000001, options={'maxfev': 100000})
                    if res.fun <= 0.0001:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        if res.fun > 0.0001:
            print("did not converge init guess is", self.pe, self.tb_mat, 'region is', self.df['region_scenario'])
        return res.x

    ## computes equilibrium price and taxes. Also computes other equilibrium values (ie consumption, production, value of exports/imports)
    ## and stores them in self. 
    def opt_tax(self):
        tax_scenario = self.tax_scenario
        varphi = self.varphi
        pes = np.append([self.pe], np.arange(0.1, 2, 0.2))
        tbs = np.append([self.tb_mat[0]], np.arange(0, 1.5, 0.2))
        props = np.append([self.tb_mat[1]], np.arange(0, 1.1, 0.5))
        self.conv = 1

        if tax_scenario['tax_sce'] == 'global':
            tbs = [0]
            opt_val = self.min_obj(props, tbs, pes)

            self.tb = 0
            self.te = varphi

        elif tax_scenario['tax_sce'] == 'PC_hybrid':
            opt_val = self.min_obj(props, tbs, pes)

            self.tb = opt_val[1]
            self.prop = opt_val[2]
            self.te = self.tb

        elif tax_scenario['tax_sce'] == 'EP_hybrid':
            pes = np.append([self.pe], np.arange(0.1, 1, 0.5))
            tbs = np.append([self.tb_mat[0]], np.arange(0, 1.5, 0.5))
            props = np.append([self.tb_mat[1]], np.arange(0.5, 1.5, 0.2))
            opt_val = self.min_obj(props, tbs, pes, [(0.01, np.inf), (0, np.inf), (0, np.inf)])

            self.tb = opt_val[1]
            self.te = opt_val[2]
            self.prop = 0

        elif tax_scenario['tax_sce'] == 'EPC_hybrid':
            opt_val = self.min_obj(props, tbs, pes)

            self.tb = opt_val[1]
            self.prop = opt_val[2]
            self.te = varphi

        elif tax_scenario['tax_sce'] == 'Unilateral' or tax_scenario['tax_sce'] == 'puretc' or tax_scenario[
            'tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EC_hybrid':
            opt_val = self.min_obj(props, tbs, pes)

            self.tb = opt_val[1]
            self.prop = opt_val[2]

            if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'puretp':
                self.te = self.tb

            elif tax_scenario['tax_sce'] == 'Unilateral' or tax_scenario['tax_sce'] == 'EC_hybrid':
                self.te = self.varphi

        elif tax_scenario['tax_sce'] == 'purete':
            res = minimize(self.find_eq, [self.pe, self.te], bounds=[(0.001, np.inf), (0, np.inf)],
                           method='nelder-mead',
                           tol=0.000001, options={'maxfev': 100000})
            if res.fun > 0.0001:
                self.conv = 0
                print("did not converge, phi is", varphi, "init guess is", self.pe, self.tb_mat, 'region is',
                      tax_scenario['tax_sce'], self.df['region_scenario'])

            self.te = res.x[1]
            tb_mat = [0, 1]
            self.tb = tb_mat[0]
            self.prop = tb_mat[1]

    # returns objective value given vector of pe, tb, and te
    def find_eq(self, p):
        pe = p[0]
        tb_mat = p[1:]
        te = self.varphi
        tax_scenario = self.tax_scenario
        if tax_scenario['tax_sce'] == 'purete':
            te = p[1]
            tb_mat = [0, 1]

        return self.comp_obj(pe, te, tb_mat)

    ## compute the objective value, currently the objective is to minimize difference between equilibrium condition
    ## which is equivalent to finding the root since we force their difference to be 0
    ## also saves optimal results in self.
    def comp_obj(self, pe, te, tb_mat):
        df = self.df
        varphi = self.varphi
        tax_scenario = self.tax_scenario
        paralist = self.paralist
        theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist
        ## compute extraction tax, and jbar's
        te, tb_mat, j_vals = comp_jbar(pe, tb_mat, te, df, tax_scenario, varphi, paralist)
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals

        # compute extraction values    
        Qe_vals = compute_qe(pe, tb_mat, te, df, paralist)
        Qe_prime, Qestar_prime, Qes, Qestars = Qe_vals
        Qeworld_prime = Qe_prime + Qestar_prime

        # compute consumption values
        cons_vals = comp_ce(pe, tb_mat, j_vals, paralist, df, tax_scenario)
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime = cons_vals
        Gestar_prime = Ceystar_prime + Cem_prime
        Cestar_prime = Ceystar_prime + Cex_prime

        # compute spending on goods
        vg_vals = comp_vg(pe, tb_mat, j_vals, cons_vals, df, tax_scenario, paralist)
        vgfin_vals = comp_vgfin(pe, tb_mat, df, tax_scenario, paralist, vg_vals, j_vals)
        Vgy_prime, Vgm_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgystar_prime = vg_vals
        Vg, Vg_prime, Vgstar, Vgstar_prime = vgfin_vals

        subsidy_ratio = 1 - ((1 - jxbar_prime) * j0_prime / ((1 - j0_prime) * jxbar_prime)) ** (1 / theta)

        ## compute value of energy used
        ve_vals = comp_ve(pe, tb_mat, tax_scenario, cons_vals)

        # compute labour used in goods production
        lg_vals = comp_lg(pe, tb_mat, df, tax_scenario, cons_vals)

        leak_vals = comp_leak(Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime, df)

        # terms that enter welfare
        delta_vals = comp_delta(pe, tb_mat, te, df, tax_scenario, varphi, paralist, Qeworld_prime, lg_vals, j_vals,
                                vgfin_vals)
        delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar = delta_vals

        chg_vals = comp_chg(df, Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime)

        # measure welfare and welfare with no emission externality
        welfare = delta_U / Vg * 100
        welfare_noexternality = (delta_U + varphi * (Qeworld_prime - df['Qeworld'])) / Vg * 100

        self.results = assign_val(pe, varphi, Qeworld_prime, ve_vals, vg_vals, vgfin_vals, delta_vals, chg_vals,
                                  leak_vals, lg_vals, subsidy_ratio, Qe_vals, welfare, welfare_noexternality, j_vals,
                                  cons_vals)

        obj_val = comp_diff(pe, tb_mat, te, paralist, varphi, tax_scenario, Qes, Qestars, Qe_prime, Qestar_prime,
                            Qeworld_prime, j_vals, cons_vals, Vgx2_prime)

        return obj_val

    ## retrieve the pandas series object containing equilibrium values (prices, taxes, consumption etc)
    def retrieve(self):
        ret = self.results
        ret['tb'] = self.tb
        ret['prop'] = self.prop
        ret['te'] = self.te
        ret['region_scenario'] = self.region
        ret['conv'] = self.conv
        return ret


## input: paralist, pe (price of energy), te (extraction tax), varphi (social cost of carbon)
##        tb_mat (tb_mat[0] = border adjustment,
##                tb_mat[1] = proportion of tax rebate on exports or extraction tax (in the case of EP_hybrid))
## output: te (extraction tax)
##         jxbar_hat, jmbar_hat, j0_hat (hat algebra for import/export threshold, 
##                                       final value obtained by multiplying by df['jxbar'] / df['jmbar'])
##         tb_mat (modify tb_mat[1] value to a default value for cases that do not use tb_mat[1])
def comp_jbar(pe, tb_mat, te, df, tax_scenario, varphi, paralist):
    # unpack parameters
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist

    ## new formulation
    Cey = df['CeHH']
    Cem = df['CeHF']
    Cex = df['CeFH']
    Ceystar = df['CeFF']

    jxbar_hat = g(pe + tb_mat[0]) ** (-theta) * Cex / (
            g(pe + tb_mat[0]) ** (-theta) * Cex + (g(pe) + tb_mat[0] * gprime(pe)) ** (-theta) * Ceystar) / df[
                    'jxbar']
    j0_hat = g(pe + tb_mat[0]) ** (-theta) * Cex / (
            g(pe + tb_mat[0]) ** (-theta) * Cex + (g(pe)) ** (-theta) * Ceystar) / df['jxbar']
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
        ve = pe + tb_mat[0]
        jmbar_hat = Cey * (g(pe) / g(ve)) ** theta / (Cey * (g(pe) / g(ve)) ** theta + Cem) / df['jmbar']
        jxbar_hat = Cex * (g(pe) / g(ve)) ** theta / (Cex * (g(pe) / g(ve)) ** theta + Ceystar) / df['jxbar']
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
        jxbar_hat = Cex * (g(pe) / g(ve)) ** theta / (Cex * (g(pe) / g(ve)) ** theta + Ceystar) / df['jxbar']

    if tax_scenario['tax_sce'] == 'EP_hybrid':
        te = tb_mat[1]
        ve = pe + tb_mat[0]
        jmbar_hat = Cey * (g(pe) / g(ve)) ** theta / (Cey * (g(pe) / g(ve)) ** theta + Cem) / df['jmbar']
        jxbar_hat = Cex * (g(pe) / g(ve)) ** theta / (Cex * (g(pe) / g(ve)) ** theta + Ceystar) / df['jxbar']

    if tax_scenario['tax_sce'] == 'EPC_hybrid':
        te = varphi
        ve = pe + tb_mat[0] - tb_mat[0] * tb_mat[1]
        jmbar_hat = 1
        jxbar_hat = Cex * (g(pe) / g(ve)) ** theta / (Cex * (g(pe) / g(ve)) ** theta + Ceystar) / df['jxbar']

    jxbar_prime = jxbar_hat * df['jxbar']
    j0_prime = j0_hat * df['jxbar']
    jmbar_prime = jmbar_hat * df['jmbar']
    j_vals = (j0_prime, jxbar_prime, jmbar_hat, jmbar_prime)

    return te, tb_mat, j_vals


## input: j0_prime, jxbar_prime, theta and sigmastar
## output: compute values for the incomplete beta functions, from 0 to j0_prime and from 0 to jxbar_prime
def incomp_betas(j0_prime, jxbar_prime, theta, sigmastar):
    def betaFun(i, theta, sigmastar):
        return i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1)

    beta_fun_val1 = quad(betaFun, 0, j0_prime, args=(theta, sigmastar))[0]
    beta_fun_val2 = quad(betaFun, 0, jxbar_prime, args=(theta, sigmastar))[0]
    return beta_fun_val1, beta_fun_val2


## input: pe (price of energy), tb_mat (tax vector), te (nomial extraction tax), df, paralist
## output: home and foreign extraction values
def compute_qe(pe, tb_mat, te, df, paralist):
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist

    Qes = []
    Qe_prime = 0
    for i in range(len(epsilonSvec)):
        petbte = pe + tb_mat[0] - te * epsilonSvec[i][1]
        if petbte < 0:
            petbte = 0
        epsS = epsilonSvec[i][0]
        prop = epsilonSvec[i][2]
        Qe = df['Qe'] * prop * petbte ** epsS
        Qe_prime += Qe
        Qes.append(Qe)

    Qestars = []
    Qestar_prime = 0
    for i in range(len(epsilonSstarvec)):
        epsSstar = epsilonSstarvec[i][0]
        prop = epsilonSstarvec[i][2]
        Qestar = df['Qestar'] * prop * pe ** epsSstar
        Qestar_prime += Qestar
        Qestars.append(Qestar)

    return Qe_prime, Qestar_prime, Qes, Qestars


## input: pe (price of energy), tb_mat (border adjustment and export rebate/extraction tax, depending on tax scenario)
##        jvals(tuple of jxbar, jmbar, j0 and their hat values (to simplify later computation))
##        paralist (tuple of user selected parameter)
##        df, tax_scenario
## output: detailed energy consumption values (home, import, export, foreign 
##         and some hat values for simplifying calculation in later steps)
def comp_ce(pe, tb_mat, jvals, paralist, df, tax_scenario):
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist
    j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    sigmatilde = (sigma - 1) / theta
    sigmastartilde = (sigmastar - 1) / theta

    # compute incomplete beta values
    beta_fun_val1, beta_fun_val2 = incomp_betas(j0_prime, jxbar_prime, theta, sigmastar)

    #### Cey, jmbar_hat = 1 if not pure tp or EP hybrid
    Cey_prime = Dstar(pe + tb_mat[0], sigma) / (Dstar(1, sigma)) * df['CeHH'] * jmbar_hat ** (1 - sigmatilde)

    #### new Cex1, Cex2
    Cex1_hat = Dstar(pe + tb_mat[0], sigmastar) / Dstar(1, sigmastar) * \
               (j0_prime / df['jxbar']) ** (1 - sigmastartilde / theta)

    const = g(pe) ** (-sigmastar) * gprime(pe + tb_mat[0]) / (g(1) ** (-sigmastar) * gprime(1))
    frac = ((1 - df['jxbar']) / df['jxbar']) ** (sigmastar / theta) * (1 - sigmastartilde)
    jterm = 1 / df['jxbar'] ** (1 - sigmastartilde / theta)
    Cex2_hat = const * frac * jterm * (beta_fun_val2 - beta_fun_val1)

    Cex1_prime = df['CeFH'] * Cex1_hat
    Cex2_prime = df['CeFH'] * Cex2_hat
    Cex_hat = Cex1_hat + Cex2_hat

    if tax_scenario['Base'] == 1:
        Cex_hat = Dstar(pe, sigmastar) / Dstar(1, sigmastar) * (jxbar_prime / df['jxbar']) ** (1 - sigmastartilde)

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        ve = pe + tb_mat[0]
        Cex_hat = Dstar(ve, sigmastar) / Dstar(1, sigmastar) * (jxbar_prime / df['jxbar']) ** (1 - sigmastartilde)

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
        Cex_hat = Dstar(ve, sigmastar) / Dstar(1, sigmastar) * (jxbar_prime / df['jxbar']) ** (1 - sigmastartilde)

    # final value for Ce^x
    Cex_prime = df['CeFH'] * Cex_hat

    # Ce^m, home imports
    Cem_hat = Dstar(pe + tb_mat[0], sigma) / Dstar(1, sigma)

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Cem_hat = Dstar(pe, sigma) / Dstar(1, sigma) * ((1 - jmbar_prime) / (1 - df['jmbar'])) ** (1 - sigmatilde)

    # final value for Ce^m
    Cem_prime = df['CeHF'] * Cem_hat

    # Ce^y*, foreign production for foreign consumption
    Ceystar_prime = Dstar(pe, sigmastar) / Dstar(1, sigmastar) * df['CeFF'] * (
            (1 - jxbar_prime) / (1 - df['jxbar'])) ** (1 - sigmatilde)

    return Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime


## input: pe (price of energy), tb_mat, jvals (import/export threshold values)
##        consvals (tuple of energy consumption values), df, tax_scenario, paralist
## output: value of goods (import, export, domestic, foreign)
def comp_vg(pe, tb_mat, jvals, consvals, df, tax_scenario, paralist):
    # unpack values from tuples
    j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = jvals
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime = consvals
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist
    sigmatilde = (sigma - 1) / theta
    sigmastartilde = (sigmastar - 1) / theta

    # BAU value of home and foreign spending on goods
    Vgx = df['CeFH'] * g(1) / gprime(1)

    Vgy_prime = g(pe + tb_mat[0]) / gprime(pe) * Cey_prime

    ## Value of exports for unilateral optimal
    Vgx1_prime = (g(pe + tb_mat[0]) / g(1)) ** (1 - sigmastar) * (j0_prime / df['jxbar']) ** (1 - sigmastartilde) * Vgx

    pterm = (g(pe) / g(1)) ** (1 - sigmastar) * Vgx
    num = (1 - j0_prime) ** (1 - sigmastartilde) - (1 - jxbar_prime) ** (1 - sigmastartilde)
    denum = df['jxbar'] * (1 - df['jxbar']) ** ((1 - sigmastar) / theta)
    Vgx2_prime = pterm * num / denum
    Vgx_hat = (Vgx1_prime + Vgx2_prime) / Vgx

    if tax_scenario['tax_sce'] != 'Unilateral':
        Vgx_hat = (g(pe) / g(1)) ** (1 - sigmastar)

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        ve = pe + tb_mat[0]
        Vgx_hat = (g(ve) / g(1)) ** (1 - sigmastar) * (jxbar_prime / df['jxbar']) ** (1 - sigmastartilde)

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
        Vgx_hat = (g(ve) / g(1)) ** (1 - sigmastar) * (jxbar_prime / df['jxbar']) ** (1 - sigmastartilde)

    # final value of home export of good    
    Vgx_prime = Vgx * Vgx_hat
    Vgm_prime = g(pe + tb_mat[0]) / gprime(pe + tb_mat[0]) * Cem_prime
    Vgystar_prime = g(pe) / gprime(pe) * Cey_prime

    return Vgy_prime, Vgm_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgystar_prime


## input: pe (price of energy), tb_mat (border adjustments), tax_scenario
##        cons_vals: tuple of energy consumption values
## output: Ve_prime, Vestar_prime (final values of home and foreign energy consumption)
def comp_ve(pe, tb_mat, tax_scenario, cons_vals):
    # unpack parameters
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime = cons_vals

    Ve_prime = (pe + tb_mat[0]) * (Cey_prime + Cem_prime)

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Ve_prime = (pe + tb_mat[0]) * Cey_prime + pe * Cem_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Ve_prime = (pe + tb_mat[0]) * Cey_prime + (pe + tb_mat[0]) * Cem_prime

    Vestar_prime = (pe + tb_mat[0]) * Cex_prime + pe * Ceystar_prime

    if tax_scenario['tax_sce'] == 'Unilateral':
        Vestar_prime = (pe + tb_mat[0]) * Cex1_prime + pe * Cex2_prime + pe * Ceystar_prime

    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Vestar_prime = pe * (Cex_prime + Ceystar_prime)

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Vestar_prime = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex_prime + pe * Ceystar_prime

    return Ve_prime, Vestar_prime


## input: pe (price of energy), tb_mat (border adjustments), df, tax_scenario
##        vg_vals (vector of value of spending on goods), paralist
## output: Vg, Vg_prime, Vgstar, Vgstar_prime (values of home and foreign total spending on goods)
##         non prime values returned to simplify later computations
def comp_vgfin(pe, tb_mat, df, tax_scenario, paralist, vg_vals, j_vals):
    # unpack parameters
    Vgy_prime, Vgm_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgystar_prime = vg_vals
    j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist

    # home spending on goods
    Vg = df['Ce'] * g(1) / gprime(1)
    Vg_prime = Vgy_prime + Vgm_prime

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        scale = g(1) / gprime(1)
        # value of home and foreign goods
        Vgy = df['CeHH'] * scale
        Vgy_prime = (g(pe + tb_mat[0]) / g(1)) ** (1 - sigma) * jmbar_hat ** (1 + (1 - sigma) / theta) * Vgy
        Vg_prime = Vgy_prime + Vgm_prime

    # foreign spending on goods
    Vgstar = df['Cestar'] * g(1) / gprime(1)
    Vgstar_prime = Vgx_prime + Vgystar_prime

    return Vg, Vg_prime, Vgstar, Vgstar_prime


## input: pe (price of energy), tb_mat (border adjustments), df, tax_scenario, cons_vals (tuple of consumptions values)
## output: Lg_prime/Lgstar_prime (labour employed in production in home and foreign)
def comp_lg(pe, tb_mat, df, tax_scenario, cons_vals):
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime = cons_vals

    ## labour employed in production in home
    Lg = 1 / k(1) * df['Ge']
    Lg_prime = 1 / k(pe + tb_mat[0]) * (Cey_prime + Cex_prime)

    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'EC_hybrid':
        Lg_prime = 1 / k(pe + tb_mat[0]) * Cey_prime + 1 / k(pe) * Cex_prime

    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        Lg_prime = 1 / k(pe + tb_mat[0]) * Cey_prime + 1 / k(pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex_prime

    ## labour employed in foreign production
    Lgstar = 1 / k(1) * df['Gestar']
    Lgstar_prime = 1 / k(pe + tb_mat[0]) * Cem_prime + 1 / k(pe) * Ceystar_prime

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid':
        Lgstar_prime = 1 / k(pe) * (Cem_prime + Ceystar_prime)

    return Lg, Lgstar, Lg_prime, Lgstar_prime


## input: pe (price of energy), tb_mat (border adjustments), te (nominal extraction tax), df, tax_scenario, varphi,
##        paralist, vgfin_vals (total spending by Home and Foreign), jvals (tuple of import/export margins)
##        Qeworld_prime, lg_vals (labour in Home and Foreign production)
## output: compute change in Le/Lestar (labour in home/foreign extraction)
##         change home utility
def comp_delta(pe, tb_mat, te, df, tax_scenario, varphi, paralist, Qeworld_prime, lg_vals, j_vals, vgfin_vals):
    Lg, Lgstar, Lg_prime, Lgstar_prime = lg_vals
    Vg, Vg_prime, Vgstar, Vgstar_prime = vgfin_vals

    # unpack parameters
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist
    j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals

    # change in labour in home/foreign extraction
    delta_Le = 0
    delta_Lestar = 0
    for i in range((len(epsilonSvec))):
        epsilonS_r = epsilonSvec[i][0]
        epsilonSstar_r = epsilonSstarvec[i][0]
        hr = epsilonSvec[i][1]

        # price faced by energy extractors
        petbte = pe + tb_mat[0] - te * hr
        if petbte < 0:
            petbte = 0

        Qe_r = epsilonSvec[i][2] * df['Qe']
        Qestar_r = epsilonSstarvec[i][2] * df['Qestar']

        delta_Le += epsilonS_r / (epsilonS_r + 1) * (petbte ** (epsilonS_r + 1) - 1) * Qe_r
        if tax_scenario['tax_sce'] != 'global':
            delta_Lestar += epsilonSstar_r / (epsilonSstar_r + 1) * (pe ** (epsilonSstar_r + 1) - 1) * Qestar_r
        else:
            delta_Lestar += epsilonSstar_r / (epsilonSstar_r + 1) * (petbte ** (epsilonSstar_r + 1) - 1) * Qestar_r

    if logit == 1:
        def Func(a, beta, gamma):
            return ((1 - gamma) * a ** beta) / (1 - gamma + gamma * a ** beta) ** 2

        petbte = pe + tb_mat[0] - te
        if petbte < 0:
            petbte = 0
        delta_Le = beta * df['Qe'] * quad(Func, 1, petbte, args=(beta, gamma))[0]
        delta_Lestar = beta * df['Qestar'] * quad(Func, 1, pe, args=(beta, gamma))[0]

    # term that is common across all delta_U calculations
    const = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) - \
            varphi * (Qeworld_prime - df['Qeworld'])

    if sigma != 1 and sigmastar != 1:
        delta_U = const + sigma / (sigma - 1) * (Vg_prime - Vg) + sigmastar / (sigmastar - 1) * (Vgstar_prime - Vgstar)
        return delta_Le, delta_Lestar, delta_U

    # values in unilateral optimal, also applies to some of the constrained policies
    delta_Vg = -math.log(g(pe + tb_mat[0]) / g(1)) * Vg
    delta_Vgstar = -(math.log(g(pe) / g(1)) + 1 / theta * math.log((1 - j0_prime) / (1 - df['jxbar']))) * Vgstar

    if tax_scenario['tax_sce'] == 'puretc' or tax_scenario['tax_sce'] == 'purete' or \
            tax_scenario['tax_sce'] == 'EC_hybrid':
        delta_Vgstar = -math.log(g(pe) / g(1)) * Vgstar

    if tax_scenario['tax_sce'] == 'puretp' or tax_scenario['tax_sce'] == 'EP_hybrid' or \
            tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        delta_Vg = -(math.log(g(pe) / g(1)) + 1 / theta * math.log((1 - jmbar_prime) / (1 - df['jmbar']))) * Vg
        delta_Vgstar = -(math.log(g(pe) / g(1)) + 1 / theta * math.log((1 - jxbar_prime) / (1 - df['jxbar']))) * Vgstar

    delta_U = delta_Vg + delta_Vgstar + const
    return delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar


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
##        Cex_prime (energy in home export), paralist, tax_scenario, df
## output: marginal leakage (-(partial Gestar / partial re) / (partial Ge / partial re))
##         for different tax scenarios.
def comp_mleak(pe, tb_mat, jvals, Cey_prime, Cem_prime, Cex_prime, Ceystar_prime, paralist, tax_scenario):
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist
    j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = jvals

    ## ve is different for puretp/EP and PC/EPC
    ve = (pe + tb_mat[0])
    if tax_scenario['tax_sce'] == 'PC_hybrid' or tax_scenario['tax_sce'] == 'EPC_hybrid':
        ve = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0])

    ## leakage for PC, EPC taxes
    djxdve = -jxbar_prime * (1 - jxbar_prime) * gprime(ve) / g(ve) * theta
    djmdve = -jmbar_prime * (1 - jmbar_prime) * gprime(ve) / g(ve) * theta

    dceydve = Dstarprime(ve, sigma) / Dstar(ve, sigma) * Cey_prime + \
              (1 + (1 - sigma) / theta) * djmdve / jmbar_prime * Cey_prime
    dcemdve = 1 / (1 - jmbar_prime) * (1 + (1 - sigma) / theta) * (-djmdve) * Cem_prime
    dcexdve = Dstarprime(ve, sigmastar) / Dstar(ve, sigmastar) * Cex_prime + \
              (1 + (1 - sigmastar) / theta) * Cex_prime / jxbar_prime * djxdve
    dceystardve = (1 + (1 - sigmastar) / theta) * Ceystar_prime * (-djxdve) / (1 - jxbar_prime)

    leak = -(dceystardve + dcemdve) / (dcexdve + dceydve)
    leakstar = -dceystardve / dcexdve

    return leak, leakstar


def comp_eps(Qes, Qe_prime, Qestars, Qestar_prime, paralist):
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist

    epsilonSstar_num = 0
    epsilonSstartilde_num = 0
    epsilonSw_num = 0
    epsilonSwtilde_num = 0

    for i in range(len(epsilonSstarvec)):
        epsilonSstar_num += epsilonSstarvec[i][0] * Qestars[i]
        epsilonSstartilde_num += epsilonSstarvec[i][0] * epsilonSstarvec[i][1] * Qestars[i]

        epsilonSw_num += epsilonSstarvec[i][0] * Qestars[i] + epsilonSvec[i][0] * Qes[i]
        epsilonSwtilde_num += epsilonSstarvec[i][0] * epsilonSstarvec[i][1] * Qestars[i] + epsilonSvec[i][0] * \
                              epsilonSvec[i][1] * Qes[i]

    epsilonSstar = epsilonSstar_num / Qestar_prime
    epsilonSstartilde = epsilonSstartilde_num / Qestar_prime
    epsilonSw = epsilonSw_num / (Qestar_prime + Qe_prime)
    epsilonSwtilde = epsilonSwtilde_num / (Qestar_prime + Qe_prime)

    return epsilonSstar, epsilonSstartilde, epsilonSw, epsilonSwtilde


## input: consval (tuple of consumption values), jvals (tuple of import/export margins),
##        Ge_prime/Gestar_prime (home/foreign production energy use),
##        Qe_prime/Qestar_prime/Qeworld_prime (home/foreign/world energy extraction),
##        Vgx2_prime (intermediate value for value of home exports),
##        pe (price of energy), tax_scenario, tb_mat (border adjustments), te (extraction tax)
##        varphi, paralist, df
## output: objective values
##         diff (difference between total consumption and extraction)
##         diff1 & diff3 (equation to compute wedge and border rebate as in table 4 in paper)
def comp_diff(pe, tb_mat, te, paralist, varphi, tax_scenario, Qes, Qestars, Qe_prime, Qestar_prime, Qeworld_prime,
              j_vals, cons_vals, Vgx2_prime):
    # unpack parameters
    j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime = cons_vals
    theta, sigma, sigmastar, epsilonSvec, epsilonSstarvec, beta, gamma, logit = paralist

    # compute marginal leakage
    leak, leak2 = comp_mleak(pe, tb_mat, j_vals, Cey_prime, Cem_prime, Cex_prime, Ceystar_prime, paralist, tax_scenario)

    # compute world energy consumption and necessary elasticities
    Ceworld_prime = Cey_prime + Cex_prime + Cem_prime + Ceystar_prime

    epsilonSstar, epsilonSstartilde, epsilonSw, epsilonSwtilde = comp_eps(Qes, Qe_prime, Qestars, Qestar_prime,
                                                                          paralist)
    # world extraction = world consumption
    diff = Qeworld_prime - Ceworld_prime
    # initialize values
    diff1 = 0
    diff2 = 0

    if tax_scenario['tax_sce'] == 'Unilateral':
        # epsilonSstar, epsilonSstartilde = comp_eps(pe, Qes, Qe_prime, Qestars, Qestar_prime, paralist)
        epsilonDstar = abs(pe * Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar))
        S = g(pe + tb_mat[0]) / gprime(pe + tb_mat[0]) * Cex2_prime - Vgx2_prime
        num = varphi * epsilonSstartilde * Qestar_prime - sigmastar * gprime(pe) * S / g(pe)
        denum = epsilonSstar * Qestar_prime + epsilonDstar * Ceystar_prime
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denum - num

    if tax_scenario['tax_sce'] == 'global':
        diff1 = Qeworld_prime - Ceworld_prime

    if tax_scenario['tax_sce'] == 'purete':
        numerator = varphi * epsilonSstartilde * Qestar_prime
        dcewdpe = abs(Dstarprime(pe, sigma) / Dstar(pe, sigma) * Cey_prime
                      + Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) * Cex_prime
                      + Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) * (Ceystar_prime + Cem_prime))
        denominator = epsilonSstar * Qestar_prime + dcewdpe * pe

        # te = varphi - consumption wedge
        diff1 = (varphi - te) * denominator - numerator

    if tax_scenario['tax_sce'] == 'puretc':
        dcestardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) * Cex_prime
                         + Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) * Ceystar_prime)

        numerator = varphi * epsilonSwtilde * Qeworld_prime
        denominator = epsilonSw * Qeworld_prime + dcestardpe * pe
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator

    if tax_scenario['tax_sce'] == 'puretp':
        numerator = varphi * epsilonSwtilde * Qeworld_prime
        ## energy price faced by home producers
        ve = pe + tb_mat[0]
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1 - jxbar_prime)
        djmbardpe = theta * gprime(pe) / g(pe) * jmbar_prime * (1 - jmbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1 + (1 - sigmastar) / theta) / (
                1 - jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1 + (1 - sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime
        dcemdpe = abs(Dstarprime(pe, sigma) / Dstar(pe, sigma) - (1 + (1 - sigma) / theta) / (
                1 - jmbar_prime) * djmbardpe) * Cem_prime
        dceydpe = abs((1 + (1 - sigma) / theta) / jmbar_prime * djmbardpe) * Cey_prime

        denominator = epsilonSw * Qeworld_prime + (dceystardpe + dcemdpe) * pe - leak * (dcexdpe + dceydpe) * pe
        # border adjustment = (1-leakage) consumption wedge
        diff1 = tb_mat[0] * denominator - (1 - leak) * numerator

    if tax_scenario['tax_sce'] == 'EC_hybrid':
        dcestardpe = abs(
            Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) * Cex_prime + Dstarprime(pe, sigmastar) / Dstar(pe,
                                                                                                             sigmastar) * Ceystar_prime)

        numerator = varphi * epsilonSstartilde * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + dcestardpe * pe
        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator

    if tax_scenario['tax_sce'] == 'EP_hybrid':
        ## energy price faced by home producers
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1 - jxbar_prime)
        djmbardpe = theta * gprime(pe) / g(pe) * jmbar_prime * (1 - jmbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1 + (1 - sigmastar) / theta) / (
                1 - jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1 + (1 - sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime
        dcemdpe = abs(Dstarprime(pe, sigma) / Dstar(pe, sigma) - (1 + (1 - sigma) / theta) / (
                1 - jmbar_prime) * djmbardpe) * Cem_prime
        dceydpe = abs((1 + (1 - sigma) / theta) / jmbar_prime * djmbardpe) * Cey_prime

        numerator = varphi * epsilonSstartilde * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + (dceystardpe + dcemdpe) * pe - leak * (dcexdpe + dceydpe) * pe

        # tp equal to (1-leakge) * consumption wedge
        diff1 = tb_mat[0] * denominator - (1 - leak) * numerator
        # requires nominal extraction tax to be equal to te + tp
        diff2 = (varphi - tb_mat[1]) * denominator - leak * numerator

    if tax_scenario['tax_sce'] == 'PC_hybrid':
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1 - jxbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1 + (1 - sigmastar) / theta) / (
                1 - jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1 + (1 - sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime

        numerator = varphi * epsilonSwtilde * Qeworld_prime
        denominator = epsilonSw * Qeworld_prime + dceystardpe * pe - leak2 * dcexdpe * pe

        diff1 = (tb_mat[0] * denominator - numerator)
        # border rebate for exports tb[1] * tb[0] = leakage * tc
        diff2 = (tb_mat[1] * tb_mat[0]) * denominator - leak2 * numerator

    if tax_scenario['tax_sce'] == 'EPC_hybrid':
        djxbardpe = theta * gprime(pe) / g(pe) * jxbar_prime * (1 - jxbar_prime)
        dceystardpe = abs(Dstarprime(pe, sigmastar) / Dstar(pe, sigmastar) - (1 + (1 - sigmastar) / theta) / (
                1 - jxbar_prime) * djxbardpe) * Ceystar_prime
        dcexdpe = abs((1 + (1 - sigmastar) / theta) / jxbar_prime * djxbardpe) * Cex_prime

        numerator = varphi * epsilonSstartilde * Qestar_prime
        denominator = epsilonSstar * Qestar_prime + dceystardpe * pe - leak2 * dcexdpe * pe

        # border adjustment = consumption wedge
        diff1 = tb_mat[0] * denominator - numerator
        # border rebate = leakage * consumption wedge
        diff2 = (tb_mat[0] * tb_mat[1]) * denominator - leak2 * numerator

    return abs(diff) + abs(diff1 * 2) + abs(diff2 * 2)


# assign values to return later
def assign_val(pe, varphi, Qeworld_prime, ve_vals, vg_vals, vgfin_vals, delta_vals, chg_vals, leak_vals, lg_vals,
               subsidy_ratio, Qe_vals, welfare, welfare_noexternality, j_vals, cons_vals):
    j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
    Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime = cons_vals
    Vgy_prime, Vgm_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgystar_prime = vg_vals
    Vg, Vg_prime, Vgstar, Vgstar_prime = vgfin_vals
    Lg, Lgstar, Lg_prime, Lgstar_prime = lg_vals
    leakage1, leakage2, leakage3 = leak_vals
    delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar = delta_vals
    Ve_prime, Vestar_prime = ve_vals
    Qe_prime, Qestar_prime, Qes, Qestars = Qe_vals
    chg_extraction, chg_production, chg_consumption, chg_Qeworld = chg_vals
    # 0 used as place holder, so formatting is consistent
    ret = pd.Series({'varphi': varphi, 'pe': pe, 'tb': 0, 'prop': 0, 'te': 0, 'jxbar_prime': jxbar_prime,
                     'jmbar_prime': jmbar_prime, 'j0_prime': j0_prime, 'Qe_prime': Qe_prime,
                     'Qestar_prime': Qestar_prime, 'Qeworld_prime': Qeworld_prime, 'Cey_prime': Cey_prime,
                     'Cex_prime': Cex_prime, 'Cem_prime': Cem_prime, 'Cex1_prime': Cex1_prime, 'Cex2_prime': Cex2_prime,
                     'Ceystar_prime': Ceystar_prime, 'Vgm_prime': Vgm_prime, 'Vgx1_prime': Vgx1_prime,
                     'Vgx2_prime': Vgx2_prime, 'Vgx_prime': Vgx_prime, 'Vg_prime': Vg_prime,
                     'Vgstar_prime': Vgstar_prime, 'Lg_prime': Lg_prime, 'Lgstar_prime': Lgstar_prime,
                     'Ve_prime': Ve_prime, 'Vestar_prime': Vestar_prime, 'delta_Le': delta_Le,
                     'delta_Lestar': delta_Lestar, 'leakage1': leakage1, 'leakage2': leakage2, 'leakage3': leakage3,
                     'chg_extraction': chg_extraction, 'chg_production': chg_production,
                     'chg_consumption': chg_consumption, 'chg_Qeworld': chg_Qeworld, 'subsidy_ratio': subsidy_ratio,
                     'delta_Vg': delta_Vg, 'delta_Vgstar': delta_Vgstar, 'welfare': welfare,
                     'welfare_noexternality': welfare_noexternality})
    for i in range(len(Qes)):
        Qe = 'Qe' + str(i + 1) + '_prime'
        Qestar = 'Qe' + str(i + 1) + 'star_prime'
        ret[Qe] = Qes[i]
        ret[Qestar] = Qestars[i]
    return ret
