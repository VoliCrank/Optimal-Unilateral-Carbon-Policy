//net install carryforward.pkg

//net sj 15-4 dm0085
***** WKW (2020): estimation of the parameter Alpha (the labor elasticity in goods manufacturing) 
***** Method: Calibrate (1-alpha)/alpha to the value of energy used in production relative to the value added.
***** Data Sources: 1. OECD TECO2; 2. OECD Input-Output Tables.
***** Execution:
	* first, take the OECD data of embodied carbon and convert CO2 measures to barrels of oil 
	* second, multiply the amount of national production to price of energy to get the value of energy embodied in production
	* last, divide value of energy by value of labor to get the (1-alpha)/alpha and solve for alpha

********* convert TECO2's unit from CO2 million tonnes to barrels of crude oil *********
*import delimited "$drp/Data/rawdata/EmbodiedCarbon_OECD.csv", clear
import delimited "rawdata/EmbodiedCarbon_OECD.csv", clear
keep cou* par* value time ind*	 /*"cou" indicates buyers/consumers in trade, "par" indicates sellers/producers*/
keep if time==2015
keep if cou=="WLD"		 		 /*keep only the total production for each country*/
//keep if strlen(ind)>=6
	gen factor=4.3e-7    		 /*one barrel of crude oil is equivalent to 0.43 metric tonnes of CO2*/
	gen oil_barrel=value/factor
	gen price=48.66 	 		 /*average closing price of West Texas Intermediate (WTI) crude oil in 2015*/
	gen energy_value=oil_barrel*price
tempfile EnergyValue 
save `EnergyValue'

********* merge the value-added data (OECD Input-Output Tables) to energy data (OECD TECO2) *********
import delimited "rawdata/ValueAdded_sector_country.csv", clear
keep if time==2015
keep if row=="LABR" 	 		 /*keep only the values for Compensation of employees*/
// collapse (sum) value,by(tosectorincolumn col cou*)
replace value=value*1000000		 /*conert unit from million dollars to dollars*/
rename value labor_value
rename col ind
rename cou par
rename country partner
merge 1:1 par* ind using `EnergyValue', keep(matched) nogen
sort ind

********* calculate alpha for three scenarios of different sector components *********
* sector definition 1: only manufacturing sectors
egen Venergy1=total(energy_value) if ind=="D10T12"|ind=="D13T15"|ind=="D16"|ind=="D17T18"|ind=="D19"|ind=="D20T21"|ind=="D22"|ind=="D23"|ind=="D24"|ind=="D25"|ind=="D26"|ind=="D27"|ind=="D28"|ind=="D29"|ind=="D30"|ind=="D31T33"
egen Vlabor1=total(labor_value) if ind=="D10T12"|ind=="D13T15"|ind=="D16"|ind=="D17T18"|ind=="D19"|ind=="D20T21"|ind=="D22"|ind=="D23"|ind=="D24"|ind=="D25"|ind=="D26"|ind=="D27"|ind=="D28"|ind=="D29"|ind=="D30"|ind=="D31T33"
gen ratio1=Venergy1/Vlabor1

* sector definition 2: manufacturing plus agriculture, electricity and construction
egen Venergy2=total(energy_value) if ind=="D01T03"|ind=="D10T12"|ind=="D13T15"|ind=="D16"|ind=="D17T18"|ind=="D19"|ind=="D20T21"|ind=="D22"|ind=="D23"|ind=="D24"|ind=="D25"|ind=="D26"|ind=="D27"|ind=="D28"|ind=="D29"|ind=="D30"|ind=="D31T33"|ind=="D35T39"|ind=="D41T43"
egen Vlabor2=total(labor_value) if ind=="D01T03"|ind=="D10T12"|ind=="D13T15"|ind=="D16"|ind=="D17T18"|ind=="D19"|ind=="D20T21"|ind=="D22"|ind=="D23"|ind=="D24"|ind=="D25"|ind=="D26"|ind=="D27"|ind=="D28"|ind=="D29"|ind=="D30"|ind=="D31T33"|ind=="D35T39"|ind=="D41T43"
gen ratio2=Venergy2/Vlabor2

* sector definition 3: manufacturing plus agriculture, electricity, construction, wholesale and retail, and transporation 
egen Venergy3=total(energy_value) if ind=="D01T03"|ind=="D10T12"|ind=="D13T15"|ind=="D16"|ind=="D17T18"|ind=="D19"|ind=="D20T21"|ind=="D22"|ind=="D23"|ind=="D24"|ind=="D25"|ind=="D26"|ind=="D27"|ind=="D28"|ind=="D29"|ind=="D30"|ind=="D31T33"|ind=="D35T39"|ind=="D41T43"|ind=="D45T47"|ind=="D49T53"
egen Vlabor3=total(labor_value) if ind=="D01T03"|ind=="D10T12"|ind=="D13T15"|ind=="D16"|ind=="D17T18"|ind=="D19"|ind=="D20T21"|ind=="D22"|ind=="D23"|ind=="D24"|ind=="D25"|ind=="D26"|ind=="D27"|ind=="D28"|ind=="D29"|ind=="D30"|ind=="D31T33"|ind=="D35T39"|ind=="D41T43"|ind=="D45T47"|ind=="D49T53"
gen ratio3=Venergy3/Vlabor3

keep ratio*
duplicates drop
keep if ratio1!=.
forv i=1(1)3{
	gen alpha`i'=1/(ratio`i'+1)
}

export excel "output/parameter_alpha.xlsx", firstrow(variables) replace
