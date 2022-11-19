********** WKW (2020): estimation of energy supply elasticity *********************************************
****** DataSource: 1.The extraction cost distribution by Asker, John, Allan CollardWexler, and Jan De Loecker (2019)
****** DataSource: 2.Energy Production amount from IEA (2014)
****** Execution: aggregate extraction cost for regions using the extraction data
*choose a subset of data: only the upper 25, 50, 75 percentile of extraction (the part of extraction with higher cost)
*regress log of production on log of cost, the resulted slope is elasticity  -----------> Ce=(Qe/D)^((1-beta)/beta)
*scenarios: 1 "US as home" 2 "EU as home" 3 "OECD (EU+US) as home" 4 "OPEC as home" 5 "ROW as home"
*In the paper, we choose scenario 3 and produce figure 5 and 6 using this script.



*********************************************************************************************
********************** Obtain Extraction Data from IEA **************************************
*********************************************************************************************

**World Energy Balances from International Energy Agency (IEA) 2010-2015
import delimited "rawdata/Energy_IEA.csv", clear

dropmiss, force
rename v4 countryname
rename v6 fueltypes
rename v8 flowtype
replace fueltypes="Crude, NGL and feedstocks" if product== "CRNGFEED"
replace countryname="China" if country=="CHINAREG"
keep if time==2014
keep if fueltypes=="Crude, NGL and feedstocks" | fueltypes=="Coal and coal products" | fueltypes=="Natural gas"| fueltypes=="Peat and peat products"| fueltypes=="Oil products"
// keep if flow=="TPES"
sort country* fuel flow

gen factor=.
replace factor=73.3/1000000 if fueltypes=="Crude, NGL and feedstocks"
replace factor =56.1/1000000 if fueltypes=="Natural gas"
replace factor =94.6/1000000 if fueltypes=="Coal and coal products"| fueltypes=="Peat and peat products"| fueltypes=="Oil products"

gen CO2Emission=factor*value

keep if fueltypes=="Crude, NGL and feedstocks"

collapse (sum) CO2Emission, by(country* flow*)
keep if flowtype=="Production"

gen Qe_eu=CO2Emission if countryname=="Memo: European Union-28"
gen Qe_us=CO2Emission if countryname=="United States"
gen Qe_opec=CO2Emission if countryname=="Memo: OPEC"
egen Qe=total(CO2Emission) if countryname=="United States" | countryname=="Memo: European Union-28"
gsort -Qe
carryforward Qe, replace
gen Qestar=CO2Emission-Qe if countryname=="World"
gsort -Qestar
carryforward Qestar, replace
gen Qe_ROW=Qestar-CO2Emission if countryname=="Memo: OPEC"

sum Qe_eu
global Qe_EU=r(mean)
sum Qe_us
global Qe_US=r(mean)
sum Qe_opec
global Qe_OPEC=r(mean)
sum Qe_ROW
global Qe_ROW=r(mean)



*********************************************************************************************
****************************** Aggregate regions ********************************************
*********************************************************************************************
import excel "rawdata/ExtractionCost_Asker.xlsx", clear sheet("raw_percentile") firstrow
rename eu* EU*

forv sce=1/5{
preserve
if `sce'==1 {
		rename US home1
		rename EU foreign1
		rename OPEC foreign2
		rename ROW foreign3
}
if `sce'==2 {
		rename EU home1
		rename US foreign1
		rename OPEC foreign2
		rename ROW foreign3
}
if `sce'==3 {
		rename EU home1
		rename US home2
		rename OPEC foreign1
		rename ROW foreign2
}

if `sce'==4 {
		rename OPEC home1
		rename US foreign1
		rename EU foreign2
		rename ROW foreign3
}	

if `sce'==5 {
		rename ROW home1
		rename US foreign1
		rename EU foreign2
		rename OPEC foreign3
}	

reshape long home foreign, i(percentile) j(order)

if `sce'==1 {
		sort foreign
		gen prod_foreign=$Qe_EU*0.05 if order==1
		replace prod_foreign=$Qe_OPEC*0.05 if order==2
		replace prod_foreign=$Qe_ROW*0.05 if order==3
		replace prod_foreign=sum(prod_foreign)
		
		sort home
		gen prod_home=$Qe_US*0.05 if order==1
		replace prod_home=sum(prod_home)

}
if `sce'==2 {
		sort foreign
		gen prod_foreign=$Qe_EU*0.05 if order==1
		replace prod_foreign=$Qe_OPEC*0.05 if order==2
		replace prod_foreign=$Qe_ROW*0.05 if order==3
		replace prod_foreign=sum(prod_foreign)
		
		sort home
		gen prod_home=$Qe_EU*0.05 if order==1
		replace prod_home=sum(prod_home)

}
if `sce'==3 {
		sort foreign
		gen prod_foreign=$Qe_OPEC*0.05 if order==1
		replace prod_foreign=$Qe_ROW*0.05 if order==2
		replace prod_foreign=sum(prod_foreign)
		
		sort home
		gen prod_home=$Qe_EU*0.05 if order==1
		replace prod_home=$Qe_US*0.05 if order==2
		replace prod_home=sum(prod_home)

}

if `sce'==4 {
		sort foreign
		gen prod_foreign=$Qe_US*0.05 if order==1
		replace prod_foreign=$Qe_EU*0.05 if order==2
		replace prod_foreign=$Qe_ROW*0.05 if order==3
		replace prod_foreign=sum(prod_foreign)
		
		sort home
		gen prod_home=$Qe_OPEC*0.05 if order==1
		replace prod_home=sum(prod_home)
}

if `sce'==5 {
		sort foreign
		gen prod_foreign=$Qe_US*0.05 if order==1
		replace prod_foreign=$Qe_EU*0.05 if order==2
		replace prod_foreign=$Qe_OPEC*0.05 if order==3
		replace prod_foreign=sum(prod_foreign)
		
		sort home
		gen prod_home=$Qe_ROW*0.05 if order==1
		replace prod_home=sum(prod_home)
}
	
	
foreach var in home foreign{
sum prod_`var'
replace prod_`var'=prod_`var'/r(max)
sort prod_`var'
gen log_prod_`var'=log(prod_`var')  
gen log_cost_`var'=log(`var')

local percentile=0.5
local perc=`percentile'*100

sum prod_`var'
gen reg_`perc'_`var'=1 if prod_`var'>=r(max)* (1 - `percentile')
reg log_prod_`var' log_cost_`var'  if reg_`perc'_`var'==1
// 	di %09.3f `_b[log_cost_`var']'
scalar b =string(_b[log_cost_`var'], "%04.2f")
global epsilonS_`var'_`sce'_`perc'=scalar(b)
display ${epsilonS_`var'_`sce'_`perc'}
predict prod_hat_`var'_`perc' if reg_`perc'_`var'!=. ,xb
replace prod_hat_`var'_`perc'=exp(prod_hat_`var'_`perc')


	sort `var'
	graph twoway line prod_hat_`var'_50 `var' || scatter  prod_`var' `var', xsc(log) ysc(log) ///
		msymbol(x)  msize(medium) mlwidth(thin) mcolor(gs7 blue)   ///
		yla(0.015625 "1/64" 0.03125 "1/32" 0.0625 "1/16"  0.125 "1/8" 0.25 "1/4" 0.5 "1/2" 1 "1", ang(h)) ///
		xla(4 8 16 32 64 128 ) ///
		ysc(range(0.0078125 1)) ///
		yti(" " "Fractions of Energy Extracted", size(small)) ///
		xti( " " "Extraction Cost ($ per barrel)", size(small)) ///
		legend( lab(1 "Upper 50th Percentile: {&epsilon}{sub:S}=  ${epsilonS_`var'_`sce'_50}") lab(2 "Data") size(vsmall) symxsize(5) cols(1) pos(5) ring(0)) ///
		graphregion(fcolor(white) lcolor(white) lwidth(vvvthick) ifcolor(white) ilcolor(white) ilwidth(vvvthick)) ///
		plotregion(lcolor(white) lwidth(vvvthick) ifcolor(white) ilcolor(white) ilwidth(vvvthick)) 	
	graph export plots/epsilonS/epsilonS_`var'_sce`sce'.eps, replace
}
restore
}

clear 
set obs 5
gen sce=_n
label define sce 1 "US as home" 2 "EU as home" 3 "OECD (EU+US) as home" 4 "OPEC as home" 5 "ROW as home"
local percentile=0.5
local perc=`percentile'*100
foreach var in home foreign{
	gen epsilonS_`var'_`perc'=.
	forv sce=1/5{
		replace epsilonS_`var'_`perc'=${epsilonS_`var'_`sce'_`perc'} if sce==`sce'
		//replace epsilonS_`var'_`perc'=${epsilonS_`var'_`sce'_`perc'} if sce ==`sce'
	}
}

label value sce sce
label list sce

export excel "output/parameter_elasticity.xlsx",firstrow(variables) replace
