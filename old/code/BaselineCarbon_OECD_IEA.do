***** WKW (2020): compute baseline values of carbon accounting (extraction, production and consumption) for seven country scenarios
***** Data Sources: 1. OECD TECO2 (embodied carbon); 2. IEA (energy extraction)



************* Part1: Aggregate carbon flows (CeHH, CeFH, CeHF, CeFF)*******************
** Data source: OECD TECO2 records embodied carbon in trades (as in domestic final demand)
** https://stats.oecd.org/Index.aspx?DataSetCode=IO_GHG_2019#
* data is for 2005-2015, we only take the year 2015
* par as source/seller country, cou as destination/buyer

* prepare the embodied carbon data 
import delimited "rawdata/EmbodiedCarbon_OECD.csv", clear
keep if industry=="TOTAL"             
keep cou* par* value time
tempfile master
save `master'

* label the EU countries 
import delimited "rawdata/EU_countrylist.csv", clear varnames(1)
rename x partner
gen parEU28=1
merge 1:m partner using `master',nogen
replace parEU28=0 if parEU28==.
tempfile master
save `master'
import delimited "rawdata/EU_countrylist.csv", clear varnames(1)
rename x country
gen couEU28=1
merge 1:m country using `master',nogen
replace couEU28=0 if couEU28==.

* label regions to differentiate from individual country
sort cou par
order cou* par* 
gen parregion=0
replace parregion=1 if strlen(par) > 3
replace parregion=1 if par=="WLD" | par=="G20"| par=="DXD"
gen couregion=0
replace couregion=1 if strlen(cou) > 3
replace couregion=1 if cou=="WLD" | cou=="G20"| cou=="DXD"
replace parEU28=. if parregion==1
replace couEU28=. if couregion==1

* begin to aggregate values for each country group
*case 1: US as Home country
bysort time: gen CeHH1=value if cou=="USA" & par=="USA"
bysort time: egen CeHF1=total(value) if cou=="USA" & parregion==0 & par!="USA"
bysort time:egen CeFH1=total(value) if par=="USA" & couregion==0 & cou!="USA"
bysort time:egen CeFF1=total(value) if par!="USA" & couregion==0 & cou!="USA" & parregion==0

*case 2: EU28 as Home country
bysort time:gen CeHH2=value if cou=="EU28" & par=="EU28"
bysort time:egen CeFF2=total(value) if couregion==0 & parregion==0 & cou!="EU28" & par!="EU28" & couEU28==0 & parEU28==0
bysort time:egen CeFH2=total(value) if par=="EU28" & couEU28==0 & couregion==0 
bysort time:egen CeHF2=total(value) if cou=="EU28" & parEU28==0 & parregion==0

*case3: OECD37 as Home country
bysort time:gen CeHH3=value if cou=="OECD" & par=="OECD"
bysort time:gen CeHF3=value if cou=="OECD" & par=="NONOECD"
bysort time:gen CeFH3=value if cou=="NONOECD" & par=="OECD"
bysort time:gen CeFF3=value if cou=="NONOECD" & par=="NONOECD"

*case 4: World as Home country
bysort time: gen CeHH4=value-130 if cou=="WLD" & par=="WLD"
gen CeHF4=30
gen CeFH4=30
gen CeFF4=70

*case 5: China as Home country
bysort time: gen CeHH5=value if cou=="CHN" & par=="CHN"
bysort time: egen CeHF5=total(value) if cou=="CHN" & parregion==0 & par!="CHN"
bysort time:egen CeFH5=total(value) if par=="CHN" & couregion==0 & cou!="CHN"
bysort time:egen CeFF5=total(value) if par!="CHN" & couregion==0 & cou!="CHN" & parregion==0


*case 6: OECD plus China as Home country
bysort time:egen CeHH6=total(value) if (cou=="OECD"|cou=="CHN") & (par=="OECD"|par=="CHN")
gsort time  -CeHH6
replace CeHH6 = CeHH6[_n-1] if missing(CeHH6) 

bysort time:gen CeHF6_1=value if cou=="OECD" & par=="CHN"
gsort time  -CeHF6_1
replace CeHF6_1 = CeHF6_1[_n-1] if missing(CeHF6_1) 
bysort time:gen CeHF6_2=value if cou=="CHN" & par=="CHN"
gsort time  -CeHF6_2
replace CeHF6_2 = CeHF6_2[_n-1] if missing(CeHF6_2) 
bysort time:egen CeHF6_3=total(value) if (cou=="CHN"|cou=="OECD") & par=="NONOECD"
gsort time  -CeHF6_3
replace CeHF6_3 = CeHF6_3[_n-1] if missing(CeHF6_3) 
gen CeHF6=CeHF6_3 - CeHF6_1 - CeHF6_2
drop CeHF6_3 CeHF6_1 CeHF6_2

bysort time:gen CeFH6_1=value if par=="OECD" & cou=="CHN"
gsort time  -CeFH6_1
replace CeFH6_1 = CeFH6_1[_n-1] if missing(CeFH6_1) 
bysort time:gen CeFH6_2=value if cou=="CHN" & par=="CHN"
gsort time  -CeFH6_2
replace CeFH6_2 = CeFH6_2[_n-1] if missing(CeFH6_2) 
bysort time:egen CeFH6_3=total(value) if (par=="CHN"|par=="OECD") & cou=="NONOECD"
gsort time  -CeFH6_3
replace CeFH6_3 = CeFH6_3[_n-1] if missing(CeFH6_3) 
gen CeFH6=CeFH6_3 - CeFH6_1 - CeFH6_2
drop CeFH6_3 CeFH6_1 CeFH6_2

bysort time:gen world=value if cou=="WLD" & par=="WLD"
gsort time  -world
replace world = world[_n-1] if missing(world) 

bysort time:gen CeFF6=world-CeHH6-CeHF6-CeFH6
drop world

*case 7: US and EU as Home country
bysort time: egen CeHH7=total(value) if (cou=="USA"|cou=="EU28") & (par=="USA"|par=="EU28")

bysort time: egen CeHF7_1=total(value) if (cou=="USA"|cou=="EU28") & parEU28==0
gsort time  -CeHF7_1
replace CeHF7_1 = CeHF7_1[_n-1] if missing(CeHF7_1) 
bysort time: egen CeHF7_2=total(value) if (cou=="USA"|cou=="EU28") & par=="USA"
gsort time  -CeHF7_2
replace CeHF7_2 = CeHF7_2[_n-1] if missing(CeHF7_2) 
gen CeHF7=CeHF7_1 - CeHF7_2 
drop CeHF7_1 - CeHF7_2 

bysort time: egen CeFH7_1=total(value) if (par=="USA"|par=="EU28") & couEU28==0
gsort time  -CeFH7_1
replace CeFH7_1 = CeFH7_1[_n-1] if missing(CeFH7_1) 
bysort time: egen CeFH7_2=total(value) if (par=="USA"|par=="EU28") & cou=="USA"
gsort time  -CeFH7_2
replace CeFH7_2 = CeFH7_2[_n-1] if missing(CeFH7_2) 
gen CeFH7=CeFH7_1 - CeFH7_2 
drop CeFH7_1 - CeFH7_2 

bysort time:egen CeFF7=total(value) if par!="USA" & parEU28==0 & cou!="USA" & couEU28==0 

* keep only the carbon flow values of the seven cases
collapse (max) C*, by(time)
reshape long  CeHH CeHF CeFH CeFF, i(time) j(case)
drop if CeHH==.
label define case 1 "US as Home" 2 "EU28 as Home" 3 "OECD37 as Home" 4 "World as Home" 5 "China as Home" 6 "OECD and China as Home" 7 "US and EU as Home"

gen Ce=CeHH+CeHF
gen Cestar=CeFF+CeFH
gen Ge=CeHH+CeFH
gen Gestar=CeFF+CeHF
gen Ceworld=Ce+Cestar
gen Geworld=Ge+Gestar

keep if time==2015
tempfile EmbodiedCO2
save `EmbodiedCO2'




************* Part 2:Aggregate energy extraction (Qe, Qestar)*******************
** Data source: IEA World Energy Balances record extraction data for different energy types: "Crude, NGL and feedstocks", "Coal and coal products", "Natural gas", "Peat and peat products", "Oil products"
*  data is for 2010-2015, we only take the year 2015
*  To convert energy unit into CO2 units, apply emission factors according to 2006 IPCC guidelines for stationary combustion

**World Energy Balances from International Energy Agency (IEA) 2010-2015
import delimited "rawdata/Energy_IEA.csv", clear
dropmiss, force
rename v4 countryname
rename v6 fueltypes
rename v8 flowtype
replace fueltypes="Crude, NGL and feedstocks" if product== "CRNGFEED"
replace countryname="China" if country=="CHINAREG"
keep if flowtype=="Production"
keep if time==2015 
keep if fueltypes=="Crude, NGL and feedstocks" |fueltypes=="Coal and coal products"|fueltypes=="Natural gas"|fueltypes=="Peat and peat products"| fueltypes=="Oil products"
sort country* fuel flow

* convert unit from TJ to million tonnes of CO2
gen factor=.
replace factor=73.3/1000000 if fueltypes=="Crude, NGL and feedstocks"
replace factor =56.1/1000000 if fueltypes=="Natural gas"
replace factor =94.6/1000000 if fueltypes=="Coal and coal products"| fueltypes=="Peat and peat products"| fueltypes=="Oil products"
gen CO2Emission=factor*value
collapse (sum) CO2Emission, by(country* flow*)

* begin to aggregate values for each country group
*case 1: US as Home country
gen Qe1=CO2Emission if countryname=="United States"
carryforward Qe1, replace
gen Qestar1=CO2Emission-Qe1 if countryname=="World"


*case 2: EU28 as Home country
gen Qe2=CO2Emission if countryname=="Memo: European Union-28"
carryforward Qe2, replace
gen Qestar2=CO2Emission-Qe2 if countryname=="World"


*case3: OECD37 as Home country
gen Qe3=CO2Emission if countryname=="Memo: OECD Total"
gen Qestar3=CO2Emission if countryname=="Memo: Non-OECD Total"

*case4: World as Home country
gen Qe4=CO2Emission if countryname=="World"
gen Qestar4=0

*case5: China as Home country
gen Qe5=CO2Emission if countryname=="People's Republic of China"
carryforward Qe5, replace
gen Qestar5=CO2Emission-Qe5 if countryname=="World"

*case6: OECD plus China as Home country
egen Qe6=total(CO2Emission) if countryname=="People's Republic of China"|countryname=="Memo: OECD Total"
carryforward Qe6, replace
gen Qestar6=CO2Emission-Qe6 if countryname=="World"

*case7: US plus EU as Home country
egen Qe7=total(CO2Emission) if countryname=="United States" | countryname=="Memo: European Union-28"
carryforward Qe7, replace
gen Qestar7=CO2Emission-Qe7 if countryname=="World"

* keep only the extraction values of the seven cases
collapse (max) Q*
generate id = _n
reshape long Qe Qestar, i(id) j(case)
drop id
gen Qeworld=Qe+Qestar
label define country_scenario 1 "US as Home" 2 "EU28 as Home" 3 "OECD37 as Home" 4 "World as Home" 5 "China as Home" 6 "OECD and China as Home" 7 "US and EU as Home"
* merge with embodied carbon data
merge 1:1 case using `EmbodiedCO2',nogen

* adjust for the difference between OECD data (embodied carbon) and IEA data (extraction) using the ratio of World total
gen ratio=Geworld/Qeworld  
replace Qe=Qe*ratio
replace Qestar=Qestar*ratio
replace Qe=32176 if case==4         /*arbitrary number for case 4: World as home*/
replace Qestar=100 if case==4
replace Qeworld=Qe+Qestar
drop ratio time
foreach var of varlist _all{		/*convert the unit from million tonnes to gigatonnes of CO2*/
replace `var'=`var'/1000
}
replace case=case*1000

rename case region_scenario
gen regionbase=region_scenario
label values regionbase country_scenario
export delimited "output/BaselineCarbon_2015.csv",replace


* manipulate it as the case with no trade in goods: trade parts (CeHF, CeFH) become 0
preserve
replace CeHH=CeHH+CeHF
replace CeHF=0
replace CeFF=CeFF+CeFH
replace CeFH=0
replace Ge=Ce
replace Gestar=Cestar
export delimited "output/BaselineCarbon_2015_noTradeinGoods.csv",replace
restore
