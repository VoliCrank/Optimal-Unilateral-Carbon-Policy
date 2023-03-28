The is the main folder of simulation code.

ces_direct_energy generates results for foreign extraction elasticity = 0.5 
while ces_direct_energy_higheps generates results for foreign extraction elasticity = 2

ces_direct_energy_renewable generates results for a world with renewable energy.

The underlying codes for all three are the same, the only difference is parameter definition. 
The only reason there are three files is that I can run them in parallel to save time.

ces_direct_energy_utils provides functions that are used to calculate results.

calibrate_alpha serves to compute the correct alpha value given rho.
Since the model uses CES production with parameters rho and alpha, we need to make sure that
the share of energy used in production is 0.15 (by assumption), hence given rho, we need to find alpha that 
satisfies that assumption.