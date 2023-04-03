import math
import numpy as np
import pandas as pd
from sympy import *
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.optimize import fsolve


# class object for finding equilibrium taxes
class taxModel:
    def __init__(self, data, tax_scenario, phi_list, model_parameters):
        self.data = data
        self.tax_scenario = tax_scenario
        self.phi_list = phi_list
        self.theta, self.sigma, self.sigmaE, self.epsilonSvec, self.epsilonSstarvec, self.rho, self.alpha = model_parameters
        self.res = []

    # iterate over all combinations of phi, tax scenarios and region scenarios to solve the model
    # pe and tb arguments are initial guesses, default to (1,0)
    def solve(self, pe_guess=1, tb_guess=0, prop_guess=0, te_guess=0):
        for index, region_data in self.data.iterrows():
            for tax in self.tax_scenario:
                pe_prev, tb_prev, prop_prev, te_prev = pe_guess, tb_guess, prop_guess, te_guess
                prices = []
                print(tax)
                # for each tax-region pair, start with some pe and tb guess
                # then for each value of phi, use the result from the previous iteration as the next guess
                for phi in self.phi_list:
                    # solve model for given phi, tax scenario, region, and guesses for pe, tb, and prop
                    price = self.solveOne(phi, tax, region_data, pe_prev, tb_prev, prop_prev, te_prev)
                    # set new guess to result of previous simulation
                    pe_prev, tb_prev, prop_prev, te_prev, conv = price
                    prices.append((phi, price))
                    print(phi)
                res = pd.Series({'region_data': region_data, 'tax': tax, 'prices': prices})
                self.res.append(res)

    def solveOne(self, phi, tax, region_data, pe, tb, prop, te):
        if tax == 'global':
            res = self.solve_obj(phi, tax, region_data)
            opt_val = res[0]

            tb = 0
            te = phi
            prop = 0

        elif tax in ['Unilateral', 'puretc', 'puretp', 'EC_hybrid']:
            res = self.solve_obj(phi, tax, region_data)
            opt_val = res[0]

            tb = opt_val[1]
            prop = 0
            te = phi

            if tax in ['puretc', 'puretp']:
                te = tb

        elif tax == 'purete':
            res = fsolve(self.te_obj, [1, 0], args=(phi, tax, region_data), full_output=True, maxfev=100000)
            opt_val = res[0]
            te = opt_val[1]


        elif tax == 'PC_hybrid':
            res = self.solve_obj(phi, tax, region_data, init_guess=[pe, tb, prop])
            opt_val = res[0]

            tb = opt_val[1]
            prop = opt_val[2]
            te = tb

        elif tax == 'EP_hybrid':
            res = self.solve_obj(phi, tax, region_data)
            opt_val = res[0]

            tb = opt_val[1]
            te = opt_val[2]
            prop = 0

        elif tax == 'EPC_hybrid':
            # opt_val = self.min_obj(props, tbs, pes, phi, tax, region_data)
            res = self.solve_obj(phi, tax, region_data, init_guess=[pe, tb, prop])
            opt_val = res[0]

            tb = opt_val[1]
            prop = opt_val[2]
            te = phi

        else:
            # tax scenario incorrect
            res = [0, 0, 0]
            opt_val = [0]

        pe = opt_val[0]
        conv = res[2]
        return pe, tb, prop, te, conv

    def solve_obj(self, phi, tax, region_data, init_guess=[1, 0, 0.5], verbose=True, second_try=True):
        res = fsolve(self.obj_system, init_guess, args=(phi, tax, region_data), full_output=True, maxfev=100000)
        if res[2] != 1:
            if verbose:
                print("did not converge, tax is", tax, "region is", region_data['regionbase'], 'phi is', phi,
                      'guess is', init_guess)
            if second_try:
                res = fsolve(self.obj_system, [1, 0.5, 0.5], args=(phi, tax, region_data), full_output=True,
                             maxfev=100000)
                if res[2] == 1 and verbose:
                    print('converged on second try')
        return res

    def te_obj(self, p, phi, tax, region_data):
        p = abs(p)
        pe = p[0]
        te = p[1]
        tb_mat = [0, 1]
        diff, diff1, diff2 = self.comp_obj(pe, te, tb_mat, phi, tax, region_data)

        return diff, diff1

    def obj_system(self, p, phi, tax, region_data):
        p = abs(p)
        pe = p[0]
        # combine tb and prop into one vector of tb_mat
        tb_mat = p[1:]
        te = phi

        diff, diff1, diff2 = self.comp_obj(pe, te, tb_mat, phi, tax, region_data)

        return diff, diff1, diff2

    # compute the objective value, currently the objective is to minimize difference between equilibrium condition
    # which is equivalent to finding the root since we force their difference to be 0
    # also saves optimal results in self.
    def comp_obj(self, pe, te, tb_mat, phi, tax, region_data):

        ## compute extraction tax, and import/export thresholds
        te, tb_mat, j_vals = self.comp_jbar(pe, tb_mat, te, region_data, tax, phi)
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals

        # compute extraction values    
        Qe_prime, Qestar_prime, Qes, Qestars = self.comp_qe(tax, pe, tb_mat, te, region_data)

        # compute consumption values
        cons_vals = self.comp_ce(pe, tb_mat, j_vals, tax, region_data)

        # compute Vgx2, spending on exported goods in region 2 for Unilateral policy
        Vgx2_prime = 0
        if tax == 'Unilateral':
            sigmatilde = (self.sigma - 1) / self.theta

            # BAU value of home and foreign spending on goods
            Vgx = region_data['Cex'] * self.g(1) / self.gprime(1)

            # counterfactual spending
            pterm = (self.g(pe) / self.g(1)) ** (1 - self.sigma) * Vgx
            num = (1 - j0_prime) ** (1 - sigmatilde) - (1 - jxbar_prime) ** (1 - sigmatilde)
            denum = region_data['jxbar'] * (1 - region_data['jxbar']) ** (-sigmatilde)
            Vgx2_prime = pterm * num / denum

        diff, diff1, diff2 = self.comp_diff(pe, tb_mat, te, phi, Qes, Qestars, Qe_prime, Qestar_prime, j_vals,
                                            cons_vals,Vgx2_prime, tax)

        return diff, diff1, diff2


    def comp_cons_eq(self, pe, te, tb_mat, phi, tax, region_data):
        return self.comp_obj(pe, te, tb_mat, phi, tax, region_data)[0]

    def retrieve(self, filename=""):
        filled_results = []
        for price_scenario in self.res:
            region_data = price_scenario['region_data']
            tax = price_scenario['tax']
            prices = price_scenario['prices']

            for (phi, (pe, tb, prop, te, conv)) in prices:
                tb_mat = [abs(tb), abs(prop)]
                res = self.comp_all(abs(pe), te, tb_mat, phi, tax, region_data)
                res['regionbase'] = region_data['regionbase']
                res['tax_sce'] = tax
                res['conv'] = conv
                filled_results.append(res)

        # convert the list of pandas series into a pandas dataframe
        df = pd.DataFrame(filled_results)

        # order columns
        cols = list(df.columns.values)
        cols.pop(cols.index('regionbase'))
        cols.pop(cols.index('tax_sce'))
        df = df[['regionbase', 'tax_sce'] + cols]
        if filename != "":
            df.to_csv(filename, header=True)
        return df

    # compute welfare
    def comp_welfare(self, tb_mat, phi, tax, region_data):

        te = phi
        if tax == 'purete':
            te = p[0]
            tb_mat = [0, 1]

        pe = fsolve(self.comp_cons_eq, [1], args = (te, tb_mat, phi, tax, region_data))[0]

        ## compute extraction tax, and import/export thresholds
        te, tb_mat, j_vals = self.comp_jbar(pe, tb_mat, te, region_data, tax, phi)

        # compute extraction values
        Qe_vals = self.comp_qe(tax, pe, tb_mat, te, region_data)
        Qe_prime, Qestar_prime, Qes, Qestars = Qe_vals
        Qeworld_prime = Qe_prime + Qestar_prime

        # compute consumption values
        cons_vals = self.comp_ce(pe, tb_mat, j_vals, tax, region_data)
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals

        # compute spending on goods
        vg_vals = self.comp_vg(pe, tb_mat, j_vals, cons_vals, tax, region_data)
        vgfin_vals = self.comp_vgfin(pe, tb_mat, vg_vals, j_vals, tax, region_data)
        Vg, Vg_prime, Vgstar, Vgstar_prime = vgfin_vals

        # compute labour used in goods production
        lg_vals = self.comp_lg(pe, tb_mat, cons_vals, tax, region_data)

        # terms that enter welfare
        delta_vals = self.comp_delta(pe, tb_mat, te, phi, Qeworld_prime, lg_vals, j_vals, vgfin_vals, cons_vals, tax,
                                     region_data)
        delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar, delta_UCed, delta_UCedstar = delta_vals

        # measure welfare and welfare with no emission externality
        welfare = delta_U / Vg * 100
        welfare_noexternality = (delta_U + phi * (Qeworld_prime - region_data['Qeworld'])) / Vg * 100
        if tax == 'global':
            welfare = delta_U / (Vg + Vgstar) * 100
            welfare_noexternality = (delta_U + phi * (Qeworld_prime - region_data['Qeworld'])) / (Vg + Vgstar) * 100

        return welfare

    # def comp_utility(self, pe, tb_mat, te, phi, Qeworld_prime, lg_vals, j_vals, vg):

    # compute all values of interest
    def comp_all(self, pe, te, tb_mat, phi, tax, region_data):

        ## compute extraction tax, and import/export thresholds
        te, tb_mat, j_vals = self.comp_jbar(pe, tb_mat, te, region_data, tax, phi)
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals

        # compute extraction values
        Qe_vals = self.comp_qe(tax, pe, tb_mat, te, region_data)
        Qe_prime, Qestar_prime, Qes, Qestars = Qe_vals
        Qeworld_prime = Qe_prime + Qestar_prime

        # compute consumption values
        cons_vals = self.comp_ce(pe, tb_mat, j_vals, tax, region_data)
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals
        Gestar_prime = Ceystar_prime + Cem_prime + Cedstar_prime
        Cestar_prime = Ceystar_prime + Cex_prime + Cedstar_prime

        # compute spending on goods
        vg_vals = self.comp_vg(pe, tb_mat, j_vals, cons_vals, tax, region_data)
        vgfin_vals = self.comp_vgfin(pe, tb_mat, vg_vals, j_vals, tax, region_data)
        Vg, Vg_prime, Vgstar, Vgstar_prime = vgfin_vals

        subsidy_ratio = 1 - ((1 - jxbar_prime) * j0_prime / ((1 - j0_prime) * jxbar_prime)) ** (1 / self.theta)

        # compute value of energy used
        ve_vals = self.comp_ve(pe, tb_mat, cons_vals, tax)

        # compute labour used in goods production
        lg_vals = self.comp_lg(pe, tb_mat, cons_vals, tax, region_data)

        # compute leakage values
        leak_vals = self.comp_leak(Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime, region_data)

        # terms that enter welfare
        delta_vals = self.comp_delta(pe, tb_mat, te, phi, Qeworld_prime, lg_vals, j_vals, vgfin_vals, cons_vals, tax,
                                     region_data)
        delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar, delta_UCed, delta_UCedstar = delta_vals

        # compute changes from the baseline
        chg_vals = self.comp_chg(Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime, region_data)

        # measure welfare and welfare with no emission externality
        welfare = delta_U / Vg * 100
        welfare_noexternality = (delta_U + phi * (Qeworld_prime - region_data['Qeworld'])) / Vg * 100
        if tax == 'global':
            welfare = delta_U / (Vg + Vgstar) * 100
            welfare_noexternality = (delta_U + phi * (Qeworld_prime - region_data['Qeworld'])) / (Vg + Vgstar) * 100

        # compute marginal leakage
        leak, leakstar = self.comp_mleak(pe, tb_mat, j_vals, cons_vals, tax)

        results = self.assign_val(pe, tb_mat, te, phi, Qeworld_prime, ve_vals, vg_vals, vgfin_vals, delta_vals,
                                  chg_vals, leak_vals, lg_vals, subsidy_ratio, Qe_vals, welfare, welfare_noexternality, j_vals,
                                  cons_vals, leak, leakstar)

        return results

    # input: pe (price of energy), te (extraction tax), phi (social cost of carbon)
    #        tb_mat (tb_mat[0] = border adjustment,
    #                tb_mat[1] = proportion of tax rebate on exports or extraction tax (in the case of EP_hybrid))
    # output: te (extraction tax)
    #         jxbar_hat, jmbar_hat, j0_hat (hat algebra for import/export threshold,
    #                                       final value obtained by multiplying by df['jxbar'] / df['jmbar'])
    #         tb_mat (modify tb_mat[1] value to a default value for cases that do not use tb_mat[1])
    def comp_jbar(self, pe, tb_mat, te, region_data, tax, phi):
        # new formulation
        Cey = region_data['Cey']
        Cem = region_data['Cem']
        Cex = region_data['Cex']
        Ceystar = region_data['Ceystar']
        jmbar = region_data['jmbar']
        jxbar = region_data['jxbar']

        # assign parameters
        theta = self.theta
        g_petb = self.g(pe + tb_mat[0])
        g_pe = self.g(pe)

        jxbar_prime = g_petb ** (-theta) * Cex / (
                g_petb ** (-theta) * Cex + (g_pe + tb_mat[0] * self.gprime(pe)) ** (-theta) * Ceystar)
        j0_prime = g_petb ** (-theta) * Cex / (g_petb ** (-theta) * Cex + g_pe ** (-theta) * Ceystar)
        jmbar_hat = 1

        if tax == 'Unilateral':
            te = phi
            tb_mat[1] = 0

        if tax == 'global':
            te = phi
            tb_mat[0] = 0
            jxbar_prime = jxbar
            jmbar_hat = 1

        if tax == 'purete':
            jxbar_prime = jxbar
            jmbar_hat = 1

        if tax == 'puretc':
            te = tb_mat[0]
            jxbar_prime = jxbar
            jmbar_hat = 1
            tb_mat[1] = 1

        if tax == 'EC_hybrid':
            te = phi
            jxbar_prime = jxbar
            jmbar_hat = 1
            tb_mat[1] = 0

        if tax == 'puretp':
            te = tb_mat[0]
            ve = pe + tb_mat[0]
            g_ve = self.g(ve)
            jmbar_hat = Cey * (g_pe / g_ve) ** theta / (Cey * (g_pe / g_ve) ** theta + Cem) / jmbar
            jxbar_prime = Cex * (g_pe / g_ve) ** theta / (Cex * (g_pe / g_ve) ** theta + Ceystar)
            tb_mat[1] = 0

        if tax == 'EP_hybrid':
            te = tb_mat[1]
            ve = pe + tb_mat[0]
            g_ve = self.g(ve)
            jmbar_hat = Cey * (g_pe / g_ve) ** theta / (Cey * (g_pe / g_ve) ** theta + Cem) / jmbar
            jxbar_prime = Cex * (g_pe / g_ve) ** theta / (Cex * (g_pe / g_ve) ** theta + Ceystar)

        if tax == 'PC_hybrid':
            te = tb_mat[0]
            ve = pe + tb_mat[0] - tb_mat[0] * tb_mat[1]
            g_ve = self.g(ve)
            jmbar_hat = 1
            jxbar_prime = Cex * (g_pe / g_ve) ** theta / (Cex * (g_pe / g_ve) ** theta + Ceystar)

        if tax == 'EPC_hybrid':
            te = phi
            ve = pe + tb_mat[0] - tb_mat[0] * tb_mat[1]
            g_ve = self.g(ve)
            jmbar_hat = 1
            jxbar_prime = Cex * (g_pe / g_ve) ** theta / (Cex * (g_pe / g_ve) ** theta + Ceystar)

        jmbar_prime = jmbar_hat * jmbar
        j_vals = (j0_prime, jxbar_prime, jmbar_hat, jmbar_prime)

        return te, tb_mat, j_vals

    # input: j0_prime, jxbar_prime, theta and sigmastar
    # output: compute values for the incomplete beta functions, from 0 to j0_prime and from 0 to jxbar_prime
    def incomp_betas(self, j0_prime, jxbar_prime):
        def beta_fun(i, theta, sigmastar):
            return i ** ((1 + theta) / theta - 1) * (1 - i) ** ((theta - sigmastar) / theta - 1)

        beta_fun_val1 = quad(beta_fun, 0, j0_prime, args=(self.theta, self.sigma))[0]
        beta_fun_val2 = quad(beta_fun, 0, jxbar_prime, args=(self.theta, self.sigma))[0]
        return beta_fun_val1, beta_fun_val2

    # input: pe (price of energy), tb_mat (tax vector), te (nominal extraction tax), region_data
    # output: home and foreign extraction values
    def comp_qe(self, tax, pe, tb_mat, te, region_data):
        epsilonSvec = self.epsilonSvec
        epsilonSstarvec = self.epsilonSstarvec
        Qes = []
        Qe_prime = 0
        for i in range(len(epsilonSvec)):
            petbte = pe + tb_mat[0] - te * epsilonSvec[i][1]
            if petbte < 0:
                petbte = 0
            epsS = epsilonSvec[i][0]
            prop = epsilonSvec[i][2]
            Qe_r = region_data['Qe'] * prop * petbte ** epsS
            Qe_prime += Qe_r
            Qes.append(Qe_r)

        Qestars = []
        Qestar_prime = 0
        for i in range(len(epsilonSstarvec)):
            epsSstar = epsilonSstarvec[i][0]
            prop = epsilonSstarvec[i][2]
            Qestar_r = region_data['Qestar'] * prop * pe ** epsSstar
            if tax == 'global':
                Qestar_r = region_data['Qestar'] * prop * (pe - te * epsilonSstarvec[i][1]) ** epsSstar
            Qestar_prime += Qestar_r
            Qestars.append(Qestar_r)

        return Qe_prime, Qestar_prime, Qes, Qestars

    # input: pe (price of energy), tb_mat (border adjustment and export rebate/extraction tax, depending on tax scenario)
    #        jvals(tuple of jxbar, jmbar, j0 and their hat values (to simplify later computation))
    #        paralist (tuple of user selected parameter)
    #        df, tax_scenario
    # output: detailed energy consumption values (home, import, export, foreign
    #         and some hat values for simplifying calculation in later steps)
    def comp_ce(self, pe, tb_mat, j_vals, tax, region_data):
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
        sigma, theta, sigmaE = self.sigma, self.theta, self.sigmaE
        sigmatilde = (sigma - 1) / theta

        # compute incomplete beta values
        beta_fun_val1, beta_fun_val2 = self.incomp_betas(j0_prime, jxbar_prime)

        # direct consumption of energy
        Ced_prime = region_data['Ced'] / (pe + tb_mat[0]) ** sigmaE
        Cedstar_prime = region_data['Cedstar'] / pe ** sigmaE

        # Cey, jmbar_hat = 1 if not pure tp or EP hybrid
        Cey_prime = self.D(pe + tb_mat[0]) / (self.D(1)) * region_data['Cey'] * jmbar_hat ** (1 - sigmatilde)

        # new Cex1, Cex2
        Cex1_hat = self.D(pe + tb_mat[0]) / self.D(1) * (j0_prime / region_data['jxbar']) ** (1 - sigmatilde)

        const = self.g(pe) ** (-sigma) * self.gprime(pe + tb_mat[0]) / (self.g(1) ** (-sigma) * self.gprime(1))
        frac = ((1 - region_data['jxbar']) / region_data['jxbar']) ** (sigma / theta) * (1 - sigmatilde)
        jterm = 1 / region_data['jxbar'] ** (1 - sigmatilde)
        Cex2_hat = const * frac * jterm * (beta_fun_val2 - beta_fun_val1)

        Cex1_prime = region_data['Cex'] * Cex1_hat
        Cex2_prime = region_data['Cex'] * Cex2_hat
        Cex_hat = Cex1_hat + Cex2_hat

        # any scenario but Unilateral
        if tax != 'Unilateral':
            Cex_hat = self.D(pe) / self.D(1) * (jxbar_prime / region_data['jxbar']) ** (1 - sigmatilde)

        if tax in ['puretp', 'EP_hybrid']:
            ve = pe + tb_mat[0]
            Cex_hat = self.D(ve) / self.D(1) * (jxbar_prime / region_data['jxbar']) ** (1 - sigmatilde)
            Ced_prime = region_data['Ced'] / ve ** sigmaE

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
            Cex_hat = self.D(ve) / self.D(1) * (jxbar_prime / region_data['jxbar']) ** (1 - sigmatilde)
            # energy price faced by Home consumers is pe + tb
            Ced_prime = region_data['Ced'] / (pe + tb_mat[0]) ** sigmaE

        # final value for Cex
        Cex_prime = region_data['Cex'] * Cex_hat

        # Cem, home imports
        Cem_hat = self.D(pe + tb_mat[0]) / self.D(1)

        if tax in ['puretp', 'EP_hybrid']:
            Cem_hat = self.D(pe) / self.D(1) * ((1 - jmbar_prime) / (1 - region_data['jmbar'])) ** (1 - sigmatilde)

        # final value for Cem
        Cem_prime = region_data['Cem'] * Cem_hat

        # Cey*, foreign production for foreign consumption
        Ceystar_prime = self.D(pe) / self.D(1) * region_data['Ceystar'] * (
                (1 - jxbar_prime) / (1 - region_data['jxbar'])) ** (1 - sigmatilde)

        return Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime

    # input: pe (price of energy), tb_mat, jvals (import/export threshold values)
    #        consvals (tuple of energy consumption values), df, tax_scenario, paralist
    # output: value of goods (import, export, domestic, foreign)
    def comp_vg(self, pe, tb_mat, j_vals, cons_vals, tax, region_data):
        # unpack values from tuples
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals
        theta, sigma = self.theta, self.sigma
        sigmatilde = (sigma - 1) / theta

        # BAU value of home and foreign spending on goods
        Vgx = region_data['Cex'] * self.g(1) / self.gprime(1)

        Vgy_prime = self.g(pe + tb_mat[0]) / self.gprime(pe + tb_mat[0]) * Cey_prime

        ## Value of exports for unilateral optimal
        Vgx1_prime = (self.g(pe + tb_mat[0]) / self.g(1)) ** (1 - sigma) * (j0_prime / region_data['jxbar']) ** (
                1 - sigmatilde) * Vgx

        pterm = (self.g(pe) / self.g(1)) ** (1 - sigma) * Vgx
        num = (1 - j0_prime) ** (1 - sigmatilde) - (1 - jxbar_prime) ** (1 - sigmatilde)
        denum = region_data['jxbar'] * (1 - region_data['jxbar']) ** ((1 - sigma) / theta)
        Vgx2_prime = pterm * num / denum
        Vgx_hat = (Vgx1_prime + Vgx2_prime) / Vgx

        if tax != 'Unilateral':
            Vgx_hat = (self.g(pe) / self.g(1)) ** (1 - sigma)

        if tax in ['puretp', 'EP_hybrid']:
            ve = pe + tb_mat[0]
            Vgx_hat = (self.g(ve) / self.g(1)) ** (1 - sigma) * (jxbar_prime / region_data['jxbar']) ** (1 - sigmatilde)

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            ve = pe + tb_mat[0] - tb_mat[1] * tb_mat[0]
            Vgx_hat = (self.g(ve) / self.g(1)) ** (1 - sigma) * (jxbar_prime / region_data['jxbar']) ** (1 - sigmatilde)

        # final value of home export of good
        Vgx_prime = Vgx * Vgx_hat
        Vgm_prime = self.g(pe + tb_mat[0]) / self.gprime(pe + tb_mat[0]) * Cem_prime
        Vgystar_prime = self.g(pe) / self.gprime(pe) * Ceystar_prime

        return Vgy_prime, Vgm_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgystar_prime

    # input: pe (price of energy), tb_mat (border adjustments), df, tax_scenario
    #        vg_vals (vector of value of spending on goods), paralist, j_vals (vector of import/export margins)
    # output: Vg, Vg_prime, Vgstar, Vgstar_prime (values of home and foreign total spending on goods)
    #         non prime values returned to simplify later computations
    def comp_vgfin(self, pe, tb_mat, vg_vals, j_vals, tax, region_data):
        # unpack parameters
        Vgy_prime, Vgm_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgystar_prime = vg_vals
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
        sigma, theta = self.sigma, self.theta

        # home spending on goods
        scale = self.g(1) / self.gprime(1)
        Vg = (region_data['Cey'] + region_data['Cem']) * scale
        Vg_prime = Vgy_prime + Vgm_prime

        if tax in ['puretp', 'EP_hybrid']:
            # value of home and foreign goods
            Vgy = region_data['Cey'] * scale
            Vgy_prime = (self.g(pe + tb_mat[0]) / self.g(1)) ** (1 - sigma) * jmbar_hat ** (
                    1 + (1 - sigma) / theta) * Vgy
            Vg_prime = Vgy_prime + Vgm_prime

        # foreign spending on goods
        Vgstar = (region_data['Ceystar'] + region_data['Cex']) * scale
        Vgstar_prime = Vgx_prime + Vgystar_prime

        return Vg, Vg_prime, Vgstar, Vgstar_prime

    # input: pe (price of energy), tb_mat (border adjustments), tax_scenario
    #        cons_vals: tuple of energy consumption values
    # output: Ve_prime, Vestar_prime (final values of home and foreign energy consumption)
    def comp_ve(self, pe, tb_mat, cons_vals, tax):
        # unpack parameters
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals

        Ve_prime = (pe + tb_mat[0]) * (Cey_prime + Cem_prime)

        if tax in ['puretp', 'EP_hybrid']:
            Ve_prime = (pe + tb_mat[0]) * Cey_prime + pe * Cem_prime

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            Ve_prime = (pe + tb_mat[0]) * Cey_prime + (pe + tb_mat[0]) * Cem_prime

        Vestar_prime = (pe + tb_mat[0]) * Cex_prime + pe * Ceystar_prime

        if tax == 'Unilateral':
            Vestar_prime = (pe + tb_mat[0]) * Cex1_prime + pe * Cex2_prime + pe * Ceystar_prime

        if tax in ['puretc', 'EC_hybrid']:
            Vestar_prime = pe * (Cex_prime + Ceystar_prime)

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            Vestar_prime = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex_prime + pe * Ceystar_prime

        return Ve_prime, Vestar_prime

    # input: pe (price of energy), tb_mat (border adjustments), df, tax_scenario, cons_vals (tuple of consumptions values)
    # output: Lg_prime/Lgstar_prime (labour employed in production in home and foreign)
    def comp_lg(self, pe, tb_mat, cons_vals, tax, region_data):
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals

        # labour employed in production in home
        Lg = 1 / self.k(1) * (region_data['Cey'] + region_data['Cex'])
        Lg_prime = 1 / self.k(pe + tb_mat[0]) * (Cey_prime + Cex_prime)

        if tax in ['puretc', 'EC_hybrid']:
            Lg_prime = 1 / self.k(pe + tb_mat[0]) * Cey_prime + 1 / self.k(pe) * Cex_prime

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            Lg_prime = 1 / self.k(pe + tb_mat[0]) * Cey_prime + 1 / self.k(
                pe + tb_mat[0] - tb_mat[1] * tb_mat[0]) * Cex_prime

        # labour employed in foreign production
        Lgstar = 1 / self.k(1) * (region_data['Cem'] + region_data['Ceystar'])
        Lgstar_prime = 1 / self.k(pe + tb_mat[0]) * Cem_prime + 1 / self.k(pe) * Ceystar_prime

        if tax in ['puretp', 'EP_hybrid']:
            Lgstar_prime = 1 / self.k(pe) * (Cem_prime + Ceystar_prime)

        return Lg, Lgstar, Lg_prime, Lgstar_prime

    # input: pe (price of energy), tb_mat (border adjustments), te (nominal extraction tax), df, tax_scenario, varphi,
    #        paralist, vgfin_vals (total spending by Home and Foreign), jvals (tuple of import/export margins)
    #        Qeworld_prime, lg_vals (labour in Home and Foreign production)
    # output: compute change in Le/Lestar (labour in home/foreign extraction)
    #         change home utility
    def comp_delta(self, pe, tb_mat, te, phi, Qeworld_prime, lg_vals, j_vals, vgfin_vals, cons_vals, tax, region_data):
        # unpack parameters
        Lg, Lgstar, Lg_prime, Lgstar_prime = lg_vals
        Vg, Vg_prime, Vgstar, Vgstar_prime = vgfin_vals
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals
        epsilonSvec, epsilonSstarvec = self.epsilonSvec, self.epsilonSstarvec
        theta, sigma, sigmaE = self.theta, self.sigma, self.sigmaE

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

            Qe_r = epsilonSvec[i][2] * region_data['Qe']
            Qestar_r = epsilonSstarvec[i][2] * region_data['Qestar']

            delta_Le += epsilonS_r / (epsilonS_r + 1) * (petbte ** (epsilonS_r + 1) - 1) * Qe_r
            if tax != 'global':
                delta_Lestar += epsilonSstar_r / (epsilonSstar_r + 1) * (pe ** (epsilonSstar_r + 1) - 1) * Qestar_r
            else:
                delta_Lestar += epsilonSstar_r / (epsilonSstar_r + 1) * (petbte ** (epsilonSstar_r + 1) - 1) * Qestar_r

        # term that is common across all delta_U calculations
        const = -delta_Le - delta_Lestar - (Lg_prime - Lg) - (Lgstar_prime - Lgstar) - phi * (
                Qeworld_prime - region_data['Qeworld'])

        # values in unilateral optimal, also applies to some of the constrained policies
        delta_Vg = -math.log(self.g(pe + tb_mat[0]) / self.g(1)) * Vg
        delta_Vgstar = -(math.log(self.g(pe) / self.g(1)) + 1 / theta * math.log(
            (1 - j0_prime) / (1 - region_data['jxbar']))) * Vgstar

        # change in direct consumption of energy that enters welfare change
        delta_UCed = -region_data['Ced'] * sigmaE * math.log(pe + tb_mat[0])
        delta_UCedstar = -region_data['Cedstar'] * sigmaE * math.log(pe)

        if tax in ['puretc', 'purete', 'EC_hybrid']:
            delta_Vgstar = -math.log(self.g(pe) / self.g(1)) * Vgstar

        if tax in ['puretp', 'EP_hybrid']:
            delta_Vg = -(math.log(self.g(pe) / self.g(1)) + 1 / theta * math.log(
                (1 - jmbar_prime) / (1 - region_data['jmbar']))) * Vg
            delta_Vgstar = -(math.log(self.g(pe) / self.g(1)) + 1 / theta * math.log(
                (1 - jxbar_prime) / (1 - region_data['jxbar']))) * Vgstar

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            delta_Vg = -(math.log(self.g(pe + tb_mat[0]) / self.g(1))) * Vg
            delta_Vgstar = -(math.log(self.g(pe) / self.g(1)) + 1 / theta * math.log(
                (1 - jxbar_prime) / (1 - region_data['jxbar']))) * Vgstar

        if sigmaE != 1:
            delta_UCed = sigmaE / (sigmaE - 1) * (Ced_prime ** ((sigmaE - 1) / sigmaE)
                                                  * region_data['Ced'] ** (1 / sigmaE) - region_data['Ced'])

            delta_UCedstar = sigmaE / (sigmaE - 1) * (Cedstar_prime ** ((sigmaE - 1) / sigmaE)
                                                      * region_data['Cedstar'] ** (1 / sigmaE) - region_data['Cedstar'])
        # assume that sigma = sigmastar
        if sigma != 1:
            delta_Vg = sigma / (sigma - 1) * (Vg_prime - Vg)
            delta_Vgstar = sigma / (sigma - 1) * (Vgstar_prime - Vgstar)

        delta_U = delta_Vg + delta_Vgstar + const + delta_UCedstar + delta_UCed

        return delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar, delta_UCed, delta_UCedstar

    # input: Qestar_prime (foregin extraction), Gestar_prime (foreign energy use in production)
    #        Cestar_prime (foregin energy consumption), Qeworld_prime (world extraction), df
    # output: returns average leakage for extraction, production and consumption
    def comp_leak(self, Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime, region_data):
        leakage1 = -(Qestar_prime - region_data['Qestar']) / (Qeworld_prime - region_data['Qeworld'])
        leakage2 = -(Gestar_prime - region_data['Gestar']) / (Qeworld_prime - region_data['Qeworld'])
        leakage3 = -(Cestar_prime - region_data['Cestar']) / (Qeworld_prime - region_data['Qeworld'])

        return leakage1, leakage2, leakage3

    ## input: df, Qestar_prime (foreign extraction), Gestar_prime (foreign energy use in production)
    ##        Cestar_prime (foregin energy consumption), Qeworld_prime (world extraction)
    ## output: compute change in extraction, production and consumption of energy relative to baseline.
    def comp_chg(self, Qestar_prime, Gestar_prime, Cestar_prime, Qeworld_prime, region_data):
        chg_extraction = Qestar_prime - region_data['Qestar']
        chg_production = Gestar_prime - region_data['Gestar']
        chg_consumption = Cestar_prime - region_data['Cestar']
        chg_Qeworld = Qeworld_prime - region_data['Qeworld']
        # print(chg_production)
        return chg_extraction, chg_production, chg_consumption, chg_Qeworld

    ## input: pe (price of energy), tb_mat (border adjustments), Cey_prime (home consumption of energy on goods produced at home)
    ##        Cex_prime (energy in home export), paralist, tax_scenario, df
    ## output: marginal leakage (-(partial Gestar / partial ve) / (partial Ge / partial ve))
    ##         for different tax scenarios.
    def comp_mleak(self, pe, tb_mat, j_vals, cons_vals, tax):
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
        theta, sigma = self.theta, self.sigma

        # ve is different for puretp/EP and PC/EPC
        ve = 0
        if tax in ['puretp', 'EP_hybrid']:
            ve = (pe + tb_mat[0])
        if tax in ['PC_hybrid', 'EPC_hybrid']:
            ve = (pe + tb_mat[0] - tb_mat[1] * tb_mat[0])
        # if not a production tax, return 0
        if ve == 0:
            return 0, 0

        ## leakage for PC, EPC taxes
        djxdve = -jxbar_prime * (1 - jxbar_prime) * self.gprime(ve) / self.g(ve) * theta
        djmdve = -jmbar_prime * (1 - jmbar_prime) * self.gprime(ve) / self.g(ve) * theta

        # consumption of energy in goods production
        dceydve = self.Dprime(ve) / self.D(ve) * Cey_prime + \
                  (1 + (1 - sigma) / theta) * djmdve / jmbar_prime * Cey_prime
        dcemdve = 1 / (1 - jmbar_prime) * (1 + (1 - sigma) / theta) * (-djmdve) * Cem_prime
        dcexdve = self.Dprime(ve) / self.D(ve) * Cex_prime + \
                  (1 + (1 - sigma) / theta) * Cex_prime / jxbar_prime * djxdve
        dceystardve = (1 + (1 - sigma) / theta) * Ceystar_prime * (-djxdve) / (1 - jxbar_prime)

        # direct consumption
        dceddve = -Ced_prime * self.sigmaE / ve

        # we don't include dcedstardve because it is equal to 0, it does not depend on ve, only on pe.
        leak = -(dceystardve + dcemdve) / (dcexdve + dceydve + dceddve)
        leakstar = -dceystardve / dcexdve

        return leak, leakstar

    def comp_eps(self, Qes, Qe_prime, Qestars, Qestar_prime):
        epsilonSvec, epsilonSstarvec = self.epsilonSvec, self.epsilonSstarvec
        epsilonSstar_num, epsilonSstartilde_num, epsilonSw_num, epsilonSwtilde_num = 0, 0, 0, 0

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

    # input: consval (tuple of consumption values), j_vals (tuple of import/export thresholds),
    #        Ge_prime/Gestar_prime (home/foreign production energy use),
    #        Qe_prime/Qestar_prime/Qeworld_prime (home/foreign/world energy extraction),
    #        Vgx2_prime (intermediate value for value of home exports),
    #        pe (price of energy), tax_scenario, tb_mat (border adjustments), te (extraction tax)
    #        varphi, paralist, df
    # output: objective values
    #         diff (difference between total consumption and extraction)
    #         diff1 & diff3 (equation to compute wedge and border rebate as in table 4 in paper)
    def comp_diff(self, pe, tb_mat, te, phi, Qes, Qestars, Qe_prime, Qestar_prime, j_vals, cons_vals, Vgx2_prime, tax):
        # unpack parameters
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals
        sigma, theta, sigmaE = self.sigma, self.theta, self.sigmaE

        # compute world energy consumption extraction
        Ceworld_prime = Cey_prime + Cex_prime + Cem_prime + Ceystar_prime + Ced_prime + Cedstar_prime
        Qeworld_prime = Qe_prime + Qestar_prime

        # compute marginal leakage
        leak, leakstar = self.comp_mleak(pe, tb_mat, j_vals, cons_vals, tax)

        # elasticity of energy supply
        # if only one energy source then tilde and non-tilde are equal
        epsilonSstar, epsilonSstartilde, epsilonSw, epsilonSwtilde = self.comp_eps(Qes, Qe_prime, Qestars, Qestar_prime)

        # first equilibrium condition: world extraction = world consumption
        diff = Qeworld_prime - Ceworld_prime
        # initialize values
        diff1, diff2 = 0, 0

        # compute values that are common to all scenarios to reduce computation
        Dprime_pe = self.Dprime(pe)
        D_pe = self.D(pe)

        if tax == 'Unilateral':
            epsilonDstar = abs(pe * Dprime_pe / D_pe)
            dcezdpe = abs(Dprime_pe / D_pe * Ceystar_prime - sigmaE * Cedstar_prime / pe)
            S = self.g(pe + tb_mat[0]) / self.gprime(pe + tb_mat[0]) * Cex2_prime - Vgx2_prime
            numerator = phi * epsilonSstartilde * Qestar_prime - sigma * self.gprime(pe) * S / self.g(pe)
            #denominator = epsilonSstar * Qestar_prime + epsilonDstar * Ceystar_prime + sigmaE * Cedstar_prime
            denominator = epsilonSstar * Qestar_prime + dcezdpe * pe
            # border adjustment = consumption wedge
            diff1 = tb_mat[0] * denominator - numerator

        if tax == 'purete':
            dcewdpe = abs(Dprime_pe / D_pe * Cey_prime
                          + Dprime_pe / D_pe * Cex_prime
                          + Dprime_pe / D_pe * (Ceystar_prime + Cem_prime)
                          + -sigmaE * Ced_prime / pe + -sigmaE * Cedstar_prime / pe)

            numerator = phi * epsilonSstartilde * Qestar_prime
            denominator = epsilonSstar * Qestar_prime + dcewdpe * pe

            # te = varphi - consumption wedge
            diff1 = (phi - te) * denominator - numerator

        if tax in ['puretc', 'EC_hybrid']:
            dcestardpe = abs(Dprime_pe / D_pe * Cex_prime
                             + Dprime_pe / D_pe * Ceystar_prime
                             + -sigmaE * Cedstar_prime / pe)

            numerator = phi * epsilonSwtilde * Qeworld_prime
            denominator = epsilonSw * Qeworld_prime + dcestardpe * pe
            if tax == 'EC_hybrid':
                numerator = phi * epsilonSstartilde * Qestar_prime
                denominator = epsilonSstar * Qestar_prime + dcestardpe * pe

            # border adjustment = consumption wedge
            diff1 = tb_mat[0] * denominator - numerator

        if tax in ['puretp', 'EP_hybrid']:
            djxbardpe = theta * self.gprime(pe) / self.g(pe) * jxbar_prime * (1 - jxbar_prime)
            djmbardpe = theta * self.gprime(pe) / self.g(pe) * jmbar_prime * (1 - jmbar_prime)
            dceystardpe = (Dprime_pe / D_pe
                           - (1 + (1 - sigma) / theta) / (1 - jxbar_prime) * djxbardpe) * Ceystar_prime
            dcexdpe = (1 + (1 - sigma) / theta) / jxbar_prime * djxbardpe * Cex_prime
            # dcexdpe = ((1 + (1 - sigmastar) / theta) / (jxbar_prime) * djxbardpe) * Cex_prime
            dcemdpe = (Dprime_pe / D_pe
                       - (1 + (1 - sigma) / theta) / (1 - jmbar_prime) * djmbardpe) * Cem_prime
            dceydpe = ((1 + (1 - sigma) / theta) / jmbar_prime * djmbardpe) * Cey_prime
            dcedstardpe = -sigmaE * Cedstar_prime / pe

            numerator = phi * epsilonSwtilde * Qeworld_prime
            denominator = (epsilonSw * Qeworld_prime - (dceystardpe + dcemdpe + dcedstardpe) * pe
                           - leak * (dcexdpe + dceydpe) * pe)
            if tax == 'EP_hybrid':
                numerator = phi * epsilonSstartilde * Qestar_prime
                denominator = (epsilonSstar * Qestar_prime - (dceystardpe + dcemdpe + dcedstardpe) * pe
                               - leak * (dcexdpe + dceydpe) * pe)
                diff2 = (phi - tb_mat[1]) * denominator - leak * numerator

            # border adjustment = (1-leakage) consumption wedge
            diff1 = tb_mat[0] * denominator - (1 - leak) * numerator

        if tax in ['PC_hybrid', 'EPC_hybrid']:
            djxbardpe = theta * self.gprime(pe) / self.g(pe) * jxbar_prime * (1 - jxbar_prime)
            dceystardpe = (Dprime_pe / D_pe
                           - (1 + (1 - sigma) / theta) / (1 - jxbar_prime) * djxbardpe) * Ceystar_prime
            dcexdpe = ((1 + (1 - sigma) / theta) / jxbar_prime * djxbardpe) * Cex_prime
            dcedstardpe = -sigmaE * Cedstar_prime / pe

            dcezstardpe = dceystardpe + dcedstardpe

            numerator = phi * epsilonSwtilde * Qeworld_prime
            denominator = epsilonSw * Qeworld_prime - dcezstardpe * pe - leakstar * dcexdpe * pe
            if tax == 'EPC_hybrid':
                numerator = phi * epsilonSstartilde * Qestar_prime
                denominator = epsilonSstar * Qestar_prime - dcezstardpe * pe - leakstar * dcexdpe * pe

            diff1 = tb_mat[0] * denominator - numerator
            # border rebate for exports tb[1] * tb[0] = leakage * tc
            diff2 = (tb_mat[1] * tb_mat[0]) * denominator - leakstar * numerator

        return diff * 100, diff1, diff2

    # assign values to return later
    def assign_val(self, pe, tb_mat, te, phi, Qeworld_prime, ve_vals, vg_vals, vgfin_vals, delta_vals, chg_vals,
                   leak_vals, lg_vals, subsidy_ratio, Qe_vals, welfare, welfare_noexternality, j_vals, cons_vals, leak,
                   leakstar):
        j0_prime, jxbar_prime, jmbar_hat, jmbar_prime = j_vals
        Cey_prime, Cex1_prime, Cex2_prime, Cex_prime, Cem_prime, Ceystar_prime, Ced_prime, Cedstar_prime = cons_vals
        Vgy_prime, Vgm_prime, Vgx1_prime, Vgx2_prime, Vgx_prime, Vgystar_prime = vg_vals
        Vg, Vg_prime, Vgstar, Vgstar_prime = vgfin_vals
        Lg, Lgstar, Lg_prime, Lgstar_prime = lg_vals
        leakage1, leakage2, leakage3 = leak_vals
        delta_Le, delta_Lestar, delta_U, delta_Vg, delta_Vgstar, delta_UCed, delta_UCedstar = delta_vals
        Ve_prime, Vestar_prime = ve_vals
        Qe_prime, Qestar_prime, Qes, Qestars = Qe_vals
        chg_extraction, chg_production, chg_consumption, chg_Qeworld = chg_vals

        ret = pd.Series(
            {'varphi': phi, 'pe': pe, 'tb': tb_mat[0], 'prop': tb_mat[1], 'te': te, 'jxbar_prime': jxbar_prime,
             'jmbar_prime': jmbar_prime, 'j0_prime': j0_prime, 'Qe_prime': Qe_prime,
             'Qestar_prime': Qestar_prime, 'Qeworld_prime': Qeworld_prime,
             'Ced_prime': Ced_prime, 'Cedstar_prime': Cedstar_prime, 'Cey_prime': Cey_prime,
             'Cex_prime': Cex_prime, 'Cem_prime': Cem_prime, 'Cex1_prime': Cex1_prime, 'Cex2_prime': Cex2_prime,
             'Ceystar_prime': Ceystar_prime, 'Vgm_prime': Vgm_prime, 'Vgx1_prime': Vgx1_prime,
             'Vgx2_prime': Vgx2_prime, 'Vgx_prime': Vgx_prime, 'Vg_prime': Vg_prime,
             'Vgstar_prime': Vgstar_prime, 'Lg_prime': Lg_prime, 'Lgstar_prime': Lgstar_prime,
             'Ve_prime': Ve_prime, 'Vestar_prime': Vestar_prime, 'delta_Le': delta_Le,
             'delta_Lestar': delta_Lestar, 'leakage1': leakage1, 'leakage2': leakage2, 'leakage3': leakage3,
             'chg_extraction': chg_extraction, 'chg_production': chg_production,
             'chg_consumption': chg_consumption, 'chg_Qeworld': chg_Qeworld, 'subsidy_ratio': subsidy_ratio,
             'delta_Vg': delta_Vg, 'delta_Vgstar': delta_Vgstar, 'delta_UCed': delta_UCed,
             'delta_UCedstar': delta_UCedstar, 'leak': leak, 'leakstar': leakstar,
             'welfare': welfare, 'welfare_noexternality': welfare_noexternality})
        for i in range(len(Qes)):
            Qe = 'Qe' + str(i + 1) + '_prime'
            Qestar = 'Qe' + str(i + 1) + 'star_prime'
            ret[Qe] = Qes[i]
            ret[Qestar] = Qestars[i]
        return ret

    # define CES production function and its derivative
    def g(self, p):
        rho = self.rho
        alpha = self.alpha
        if rho == 0:
            return alpha ** (-alpha) * (1 - alpha) ** (-(1 - alpha)) * p ** alpha
        else:
            t1 = (1 - alpha) ** (1 / (1 - rho))
            t2 = alpha ** (1 / (1 - rho)) * p ** (-rho / (1 - rho))
            return (t1 + t2) ** (-(1 - rho) / rho)

    def gprime(self, p):
        rho = self.rho
        alpha = self.alpha
        if rho == 0:
            return (alpha / (1 - alpha)) ** (1 - alpha) * p ** (-(1 - alpha))
        else:
            t1 = (1 - alpha) ** (1 / (1 - rho))
            t2 = alpha ** (1 / (1 - rho)) * p ** (-rho / (1 - rho))
            coef = alpha ** (1 / (1 - rho)) * p ** (-rho / (1 - rho) - 1)
            return (t1 + t2) ** (-(1 - rho) / rho - 1) * coef

    def k(self, p):
        return self.gprime(p) / (self.g(p) - p * self.gprime(p))

    # D(p, sigmastar) corresponds to D(p) in paper
    def D(self, p):
        return self.gprime(p) * self.g(p) ** (-self.sigma)

    def Dprime(self, p):
        x = symbols('x')
        return diff(self.D(x), x).subs(x, p)
