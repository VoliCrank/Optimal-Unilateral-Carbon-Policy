#This file generates all figures for the paper

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as fm
import palettable


mpl.__version__
mpl.rcParams.update({'font.size': 7})
mpl.rcParams['axes.grid'] = True
mpl.rcParams.update({'axes.grid.axis':'y','grid.color':'#949494','grid.linewidth':0.2})
plt.locator_params(axis='y', nbins=5)   # y-axis
# plt.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.family"] = "Linux Libertine"
fontpath = "/Users/bellayao/Library/Fonts/LinLibertine_R.ttf" #For Bella's local
prop = fm.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['lines.linewidth']=1


################################################################################
#######Fig 1:# Outcomes of optimal tax in OECD (two elasticities)#############
################################################################################
df = pd.read_csv('output/output_case3.csv');
dff = pd.read_csv('output/output_case3_D_2.csv');
df['eff_te']=df['te']-df['tb']
dff['eff_te']=dff['te']-dff['tb']

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.set_position([.1, .6, .2, .35])
ax2.set_position([.425, .6, .2, .35])
ax3.set_position([.75, .6, .2, .35])
ax4.set_position([.1, .15, .2, .35])
ax5.set_position([.425, .15, .2, .35])
ax6.set_position([.75, .15, .2, .35])


for ax in fig.get_axes():
    ax.set_xlim(0,2)
    ax.spines["top"].set_visible(False)   
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    

df1=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==3)]
l1,=ax1.plot(df1['varphi'],df1['Qeworld_prime'], color='C0',linewidth=1)
ax2.plot(df1['varphi'],df1['pe'], color='C0',linewidth=1)
ax3.plot(df1['varphi'],df1['jxbar_prime'], color='C0',linewidth=1)
ax4.plot(df1['varphi'],df1['welfare'], color='C0',linewidth=1)
ax5.plot(df1['varphi'],df1['tb'], color='C0',linewidth=1)
ax5.plot(df1['varphi'],df1['eff_te'], linestyle='dotted', color='C0',linewidth=1)
ax6.plot(df1['varphi'],df1['subsidy_ratio'], color='C0',linewidth=1)

df1=dff[(dff['tax_sce']=='Unilateral') & (dff['region_scenario']==3)]
l2,=ax1.plot(df1['varphi'],df1['Qeworld_prime'], color='C1',linewidth=0.5,linestyle='dashed')
ax2.plot(df1['varphi'],df1['pe'], color='C1',linewidth=0.5,linestyle='dashed')
ax3.plot(df1['varphi'],df1['jxbar_prime'], color='C1',linewidth=0.5,linestyle='dashed')
ax4.plot(df1['varphi'],df1['welfare'], color='C1',linewidth=0.5,linestyle='dashed')
ax5.plot(df1['varphi'],df1['tb'], color='C1',linewidth=0.5,linestyle='dashed')
ax5.plot(df1['varphi'],df1['eff_te'], color='C1',linewidth=0.5,linestyle='dashdot')
ax6.plot(df1['varphi'],df1['subsidy_ratio'], color='C1',linewidth=0.5,linestyle='dashed')

ax4.set_xlabel('Marginal harm')
ax5.set_xlabel('Marginal harm')
ax6.set_xlabel('Marginal harm')
ax1.set_ylabel('Global emissions \n(gigatonnes of CO2)')
ax2.set_ylabel('Energy price')
ax3.set_ylabel('Export margin')
ax4.set_ylabel('Change in welfare')
ax5.set_ylabel('Tax rates')
ax6.set_ylabel('Maximum export subsidy \nrelative to cost')

ax1.set_ylim(24, 34)
ax2.set_ylim(0.9, 1.1)
ax3.set_ylim(0.045, 0.065)
ax4.set_ylim(0, 10)
ax5.set_ylim(0, 2)
ax6.set_ylim(0, 0.2)

ax1.set_yticks([24,26,28,30,32])
ax2.set_yticks([0.9,0.95,1.0,1.05])
ax3.set_yticks([0.045, 0.050, 0.055, 0.060])
ax4.set_yticks([0,2,4,6,8])
ax5.set_yticks([0,0.5,1,1.5])
ax6.set_yticks([0,0.05,0.1,0.15])

ax5.legend( (r"$t_b (\epsilon_S^*=0.5)$", "$t_e (\epsilon_S^*=0.5)$", "$t_b (\epsilon_S^*=2)$", "$t_e (\epsilon_S^*=2)$"),fontsize=6, loc='upper left', 
            shadow=False, frameon=False)

ax4.legend(handles = [l1,l2] , labels=[r"$\epsilon_S=0.5$, $\epsilon_S^*=0.5$","$\epsilon_S=0.5$, $\epsilon_S^*=2$"],
            loc='upper left', bbox_to_anchor=(1, -0.25),fancybox=True, shadow=False, ncol=4, frameon=False)

plt.savefig('plots/fig1.eps', format='eps')
plt.savefig('plots/fig1.pdf', format='PDF')


##############################################################################
### Fig 2: two PPFs with two elasticities (EU, EU/US, OECD, OECD/China, World) ####
##############################################################################

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.set_position([.08, .32, .37, .52])
ax2.set_position([.58, .32, .37, .52])

x=0
for ax in fig.get_axes():
    ax.set_xlim(-10,0)
    ax.set_ylim(0,71)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.set_xlabel('Change in total consumption \n(% of initial goods consumption)')
    
    x=x+1
    if x==1:
        df = pd.read_csv('output/output_case3.csv'); #elasticity (0.5,0.5)
    elif x==2:
        df = pd.read_csv('output/output_case3_D_2.csv'); #elasticity (0.5,2)

    ##EU
    df1=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==2)] 
    df1['Qeworld_chg']=-(df1['Qeworld_prime']-32.2760)/32.2760*100;
    l1, = ax.plot(df1['welfare_noexternality'],df1['Qeworld_chg'],linestyle='solid')
    dot=df1[df1['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
        
    
    ##EU/US
    df2=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==7)] 
    df2['Qeworld_chg']=-(df2['Qeworld_prime']-32.2760)/32.2760*100;
    l2, = ax.plot(df2['welfare_noexternality'],df2['Qeworld_chg'],linestyle='dotted')
    dot=df2[df2['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
       
    
    ##OECD
    df3=df[(df['tax_sce']=='Unilateral')  & (df['region_scenario']==3)]
    df3['Qeworld_chg']=-(df3['Qeworld_prime']-32.2760)/32.2760*100;
    l3, = ax.plot(df3['welfare_noexternality'],df3['Qeworld_chg'],linestyle='dashed')
    dot=df3[df3['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
       
    
    ##OECD/China
    df4=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==6)]
    df4['Qeworld_chg']=-(df4['Qeworld_prime']-32.2760)/32.2760*100;
    l4, = ax.plot(df4['welfare_noexternality'],df4['Qeworld_chg'],linestyle='dashdot')
    dot=df4[df4['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
       
    
    ##World
    df5=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==4)] 
    df5['Qeworld_chg']=-(df5['Qeworld_prime']-32.2760)/32.2760*100;
    l5, = ax.plot(df5['welfare_noexternality'],df5['Qeworld_chg'],linestyle='dashed',dashes = (5,2))
    dot=df5[df5['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
       

ax1.set_ylabel('Global emissions reductions (% of BAU)')
ax1.set_title(r"$\epsilon_S=0.5$, $\epsilon_S^\ast=0.5$", fontsize=6)
ax2.set_title(r"$\epsilon_S=0.5$, $\epsilon_S^\ast=2$", fontsize=6)

ax1.legend(handles = [l1,l2,l3,l4,l5] , labels=['EU', 'EU+US','OECD','OECD+China','World'],
            loc='upper left', bbox_to_anchor=(0.1, -0.27),fancybox=True, shadow=False, ncol=5,
            frameon=False)
# 

plt.savefig('plots/fig2.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0.0)
plt.savefig('plots/fig2.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.0)



##############################################################################
######### Fig3: two PPFs with twasticities (7 taxes)###########################
##############################################################################

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_position([.1, .4, .35, .45])
ax2.set_position([.6, .4, .35, .45])

x=0
for ax in fig.get_axes():
    ax.set_xlim(-10,0)
    ax.set_ylim(0,31)
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)

    x=x+1
    if x==1:
        df = pd.read_csv('output/output_case3.csv');
    elif x==2:
        df = pd.read_csv('output/output_case3_D_2.csv');

    Qeworld=df['Qeworld_prime'][1]
    df['chg_Qeworld']=df['chg_Qeworld']/Qeworld*100
    
    ##puretp
    df1=df[(df['tax_sce']=='puretp') & (df['region_scenario']==3)] 
    df1['Qeworld_chg']=-(df1['Qeworld_prime']-32.2760)/32.2760*100;
    l1, = ax.plot(df1['welfare_noexternality'],df1['Qeworld_chg'],linestyle='dashed',dashes = (5,1))
    dot=df1[df1['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
       
    ##purete
    df2=df[(df['tax_sce']=='purete') & (df['region_scenario']==3)] 
    df2['Qeworld_chg']=-(df2['Qeworld_prime']-32.2760)/32.2760*100;
    l2, = ax.plot(df2['welfare_noexternality'],df2['Qeworld_chg'],linestyle='dashed',dashes = (8,1))
    dot=df2[df2['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
    
    ##puretc
    df3=df[(df['tax_sce']=='puretc') & (df['region_scenario']==3)] 
    df3['Qeworld_chg']=-(df3['Qeworld_prime']-32.2760)/32.2760*100;
    l3, = ax.plot(df3['welfare_noexternality'],df3['Qeworld_chg'],linestyle='dashed',dashes = (5,3))
    dot=df3[df3['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
    
    ##production/consumption
    df4=df[(df['tax_sce']=='PC_hybrid') & (df['region_scenario']==3)] 
    df4['Qeworld_chg']=-(df4['Qeworld_prime']-32.2760)/32.2760*100;
    l4, = ax.plot(df4['welfare_noexternality'],df4['Qeworld_chg'],linestyle='dotted')
    dot=df4[df4['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
    
    ##extracion/production
    df5=df[(df['tax_sce']=='EP_hybrid')  & (df['region_scenario']==3) &(df['varphi']<=2.6)] 
    df5['Qeworld_chg']=-(df5['Qeworld_prime']-32.2760)/32.2760*100;
    l5, = ax.plot(df5['welfare_noexternality'],df5['Qeworld_chg'],linestyle='dashed')
    dot=df5[df5['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
  
    # #extraction/consumption
    df6=df[(df['tax_sce']=='EC_hybrid') & (df['region_scenario']==3)]
    df6['Qeworld_chg']=-(df6['Qeworld_prime']-32.2760)/32.2760*100;
    l6, = ax.plot(df6['welfare_noexternality'],df6['Qeworld_chg'],linestyle='dashdot')
    dot=df6[df6['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 
    
    ##unilateral optimal
    df7=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==3)] 
    df7['Qeworld_chg']=-(df7['Qeworld_prime']-32.2760)/32.2760*100;
    l7, = ax.plot(df7['welfare_noexternality'],df7['Qeworld_chg'],linestyle='solid')
    dot=df7[df7['varphi']==2] #puretp
    ax.plot(dot['welfare_noexternality'],dot['Qeworld_chg'], marker='x', markersize=3, color='red', label='point') 


ax1.set_ylabel('Global emissions reductions (% of BAU)')
ax1.set_xlabel('Change in total consumption \n(% of initial goods consumption)')
ax2.set_xlabel('Change in total consumption \n(% of initial goods consumption)')
ax1.set_title(r"$\epsilon_S=0.5$, $\epsilon_S^*=0.5$", fontsize=6)
ax2.set_title(r"$\epsilon_S=0.5$, $\epsilon_S^*=2$", fontsize=6)


ax1.legend(handles = [l1,l2,l3,l4,l5,l6,l7] , labels=['production tax', 'extraction tax','consumption tax','production-consumption',
                                                'extraction-production','extraction-consumption','optimal'],
            loc='upper left', bbox_to_anchor=(-0.1, -0.25),fancybox=True, shadow=False, ncol=4, frameon=False)

plt.savefig('plots/fig3.pdf', format='pdf', bbox_inches = 'tight')
plt.savefig('plots/fig3.eps', format='eps', bbox_inches = 'tight', pad_inches = 0.0)



######################################################
############## Fig 4:Price of Energy #############
######################################################
df = pd.read_csv('output/output_case3.csv');
fig, ax1 =plt.subplots()

df1=df[(df['tax_sce']=='puretp') & (df['region_scenario']==3)]
ax1.plot(df1['varphi'],df1['pe'],linestyle='dashed',dashes = (5,1), label = 'production tax')

df2=df[(df['tax_sce']=='purete') & (df['region_scenario']==3)]
ax1.plot(df2['varphi'],df2['pe'],linestyle='dashed',dashes = (8,1), label = 'extraction tax')

df3=df[(df['tax_sce']=='puretc') & (df['region_scenario']==3)]
ax1.plot(df3['varphi'],df3['pe'],linestyle='dashed',dashes = (5,3),label = 'consumption tax')

df4=df[(df['tax_sce']=='PC_hybrid') & (df['region_scenario']==3)]
ax1.plot(df4['varphi'],df4['pe'],linestyle='dotted',label = 'production-consumption')

df5=df[(df['tax_sce']=='EP_hybrid') & (df['region_scenario']==3)]
ax1.plot(df5['varphi'],df5['pe'],linestyle='dashed',label = 'extraction-production')

df6=df[(df['tax_sce']=='EC_hybrid') & (df['region_scenario']==3)]
ax1.plot(df6['varphi'],df6['pe'],linestyle='dashdot', label = 'extraction-consumption')

df7=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==3)]
ax1.plot(df7['varphi'],df7['pe'],linestyle='solid',label = 'optimal')


ax1.spines["top"].set_visible(False)    
ax1.spines["bottom"].set_visible(True)    
ax1.spines["right"].set_visible(False)    
ax1.spines["left"].set_visible(False)

ax1.set_xticks([0,0.5,1,1.5,2])
ax1.set_position([.1, .2, .75,.75])
ax1.set_xlabel('Marginal harm')
ax1.set_xlim(0,2)
ax1.set_ylabel('Price of energy')
ax1.set_title( 'Figure 3: Effects on the price of energy',loc='left', pad=48)

# ax1.legend(loc='upper left', bbox_to_anchor=(-0.03, -0.25),fancybox=True, shadow=False, ncol=5)
ax1.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.535, -0.28), fontsize = 7, frameon=False)
plt.savefig('plots/price_of_energy.eps', format='eps' )
plt.savefig('plots/price_of_energy.pdf', format='pdf' )
plt.show()

##############################################################################
################Fig 5: location effects##############################
##############################################################################
df = pd.read_csv('output/output_case3.csv');
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for ax in fig.get_axes():
    ax.set_xlim(0,2)
    ax.spines["top"].set_visible(False)   
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.locator_params(axis='y', nbins=5)   # y-axis
    ax.locator_params(axis='x', nbins=5)   # y-axis

ax3.set_xlabel('Marginal harm')
ax4.set_xlabel('Marginal harm')
ax1.set_ylabel('Percent change from BAU')
ax3.set_ylabel('Percent change from BAU')
ax1.set_position([.1, .6, .35, .35])
ax2.set_position([.6, .6, .35, .35])
ax3.set_position([.1, .15, .35, .35])
ax4.set_position([.6, .15, .35, .35])
 

#
df1=df[(df['tax_sce']=='purete') & (df['region_scenario']==3)]
l1, = ax1.plot(df1['varphi'],df1['chg_extraction'],linestyle='dashdot')
ax2.plot(df1['varphi'],df1['chg_production'],linestyle='dashdot')
ax3.plot(df1['varphi'],df1['chg_consumption'],linestyle='dashdot')
ax4.plot(df1['varphi'],df1['chg_Qeworld'],linestyle='dashdot')

#
df2=df[(df['tax_sce']=='puretp') & (df['region_scenario']==3)]
l2, = ax1.plot(df2['varphi'],df2['chg_extraction'],linestyle='dashed',dashes = (8,5))
ax2.plot(df2['varphi'],df2['chg_production'],linestyle='dashed',dashes = (8,5))
ax3.plot(df2['varphi'],df2['chg_consumption'],linestyle='dashed',dashes = (8,5))
ax4.plot(df2['varphi'],df2['chg_Qeworld'],linestyle='dashed',dashes = (8,5))


#
df3=df[(df['tax_sce']=='PC_hybrid') & (df['region_scenario']==3)]
l3, = ax1.plot(df3['varphi'],df3['chg_extraction'],linestyle='dotted')
ax2.plot(df3['varphi'],df3['chg_production'],linestyle='dotted')
ax3.plot(df3['varphi'],df3['chg_consumption'],linestyle='dotted')
ax4.plot(df3['varphi'],df3['chg_Qeworld'],linestyle='dotted')

#
df4=df[(df['tax_sce']=='EP_hybrid') & (df['region_scenario']==3)]
l4, = ax1.plot(df4['varphi'],df4['chg_extraction'],linestyle='dashed')
ax2.plot(df4['varphi'],df4['chg_production'],linestyle='dashed')
ax3.plot(df4['varphi'],df4['chg_consumption'],linestyle='dashed')
ax4.plot(df4['varphi'],df4['chg_Qeworld'],linestyle='dashed')

#
df5=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==3)]
l5, = ax1.plot(df5['varphi'],df5['chg_extraction'],linestyle='solid')
ax2.plot(df5['varphi'],df5['chg_production'],linestyle='solid')
ax3.plot(df5['varphi'],df5['chg_consumption'],linestyle='solid')
ax4.plot(df5['varphi'],df5['chg_Qeworld'],linestyle='solid')

ax1.set_title('Extraction',loc='left',fontsize = 8, pad=3)
ax1.set_ylim(-10,15)
ax1.set_yticks([-10,-5, 0, 5,10, 15])
ax2.set_title('Production',loc='left',fontsize = 8, pad=3)
ax2.set_ylim(-20,20)
ax2.set_yticks([-20,-10 , 0, 10, 20])
ax3.set_title('Consumption',loc='left',fontsize = 8, pad=3)
ax3.set_ylim(-20,20)
ax2.set_yticks([-20,-10 , 0, 10, 20])
ax4.set_title('Global Emissions',loc='left',fontsize = 8, pad=3)
ax4.set_ylim(-25, 2)
ax4.set_yticks([-20., -15, -10, -5, 0])

ax4.legend(handles = [l1,l2,l3,l4,l5] , 
            labels=['extraction tax', 'production tax','production-consumption', 
                    'extraction-production','optimal'],
            loc='upper left', bbox_to_anchor=(-1.4, -0.21),
            fancybox=True, shadow=False, ncol=3, frameon=False)

plt.savefig('plots/leakage.pdf', format='pdf')
plt.savefig('plots/leakage.eps', format='eps')

plt.show()

##############################################################################
################Fig 5: location effects with seven taxes##############################
##############################################################################
df = pd.read_csv('output/output_case3_D_2.csv');
Qestar=df['Qestar_prime'][1]    #BAU values
Qeworld=df['Qeworld_prime'][1]
Cestar=df['CeFF_prime'][1]+df['CeFH_prime'][1]
Gestar=df['CeHF_prime'][1]+df['CeFF_prime'][1]
df['chg_extraction']=df['chg_extraction']/Qestar*100  #change from absolute change to percent change
df['chg_production']=df['chg_production']/Gestar*100
df['chg_consumption']=df['chg_consumption']/Cestar*100
df['chg_Qeworld']=df['chg_Qeworld']/Qeworld*100
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
for ax in fig.get_axes():
    ax.set_xlim(0,2)
    ax.spines["top"].set_visible(False)   
    ax.spines["bottom"].set_visible(True)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.locator_params(axis='y', nbins=5)   # y-axis
    ax.locator_params(axis='x', nbins=5)   # y-axis

ax3.set_xlabel('Marginal harm')
ax4.set_xlabel('Marginal harm')
ax1.set_ylabel('Percent change from BAU')
ax3.set_ylabel('Percent change from BAU')
ax1.set_position([.1, .6, .35, .35])
ax2.set_position([.6, .6, .35, .35])
ax3.set_position([.1, .15, .35, .35])
ax4.set_position([.6, .15, .35, .35])
 

#
df1=df[(df['tax_sce']=='puretp') & (df['region_scenario']==3)]
l1, = ax1.plot(df1['varphi'],df1['chg_extraction'],linestyle='dashed',dashes = (5,1))
ax2.plot(df1['varphi'],df1['chg_production'],linestyle='dashed',dashes = (5,1))
ax3.plot(df1['varphi'],df1['chg_consumption'],linestyle='dashed',dashes = (5,1))
ax4.plot(df1['varphi'],df1['chg_Qeworld'],linestyle='dashed',dashes = (5,1))

#
df2=df[(df['tax_sce']=='purete') & (df['region_scenario']==3)]
l2, = ax1.plot(df2['varphi'],df2['chg_extraction'],linestyle='dashed',dashes = (8,1))
ax2.plot(df2['varphi'],df2['chg_production'],linestyle='dashed',dashes = (8,1))
ax3.plot(df2['varphi'],df2['chg_consumption'],linestyle='dashed',dashes = (8,1))
ax4.plot(df2['varphi'],df2['chg_Qeworld'],linestyle='dashed',dashes = (8,1))

#
df3=df[(df['tax_sce']=='puretc') & (df['region_scenario']==3)]
l3, = ax1.plot(df3['varphi'],df3['chg_extraction'],linestyle='dashed', dashes=(5,3))
ax2.plot(df3['varphi'],df3['chg_production'],linestyle='dashed', dashes=(5,3))
ax3.plot(df3['varphi'],df3['chg_consumption'],linestyle='dashed', dashes=(5,3))
ax4.plot(df3['varphi'],df3['chg_Qeworld'],linestyle='dashed', dashes=(5,3))

#
df4=df[(df['tax_sce']=='PC_hybrid') & (df['region_scenario']==3)]
l4, = ax1.plot(df4['varphi'],df4['chg_extraction'],linestyle='dotted')
ax2.plot(df4['varphi'],df4['chg_production'],linestyle='dotted')
ax3.plot(df4['varphi'],df4['chg_consumption'],linestyle='dotted')
ax4.plot(df4['varphi'],df4['chg_Qeworld'],linestyle='dotted')

#
df5=df[(df['tax_sce']=='EP_hybrid') & (df['region_scenario']==3)]
l5, = ax1.plot(df5['varphi'],df5['chg_extraction'],linestyle='dashed')
ax2.plot(df5['varphi'],df5['chg_production'],linestyle='dashed')
ax3.plot(df5['varphi'],df5['chg_consumption'],linestyle='dashed')
ax4.plot(df5['varphi'],df5['chg_Qeworld'],linestyle='dashed')


#
df6=df[(df['tax_sce']=='EC_hybrid') & (df['region_scenario']==3)]
l6, = ax1.plot(df6['varphi'],df6['chg_extraction'],linestyle='dashdot')
ax2.plot(df6['varphi'],df6['chg_production'],linestyle='dashdot')
ax3.plot(df6['varphi'],df6['chg_consumption'],linestyle='dashdot')
ax4.plot(df6['varphi'],df6['chg_Qeworld'],linestyle='dashdot')

#
df7=df[(df['tax_sce']=='Unilateral') & (df['region_scenario']==3)]
l7, = ax1.plot(df7['varphi'],df7['chg_extraction'],linestyle='solid')
ax2.plot(df7['varphi'],df7['chg_production'],linestyle='solid')
ax3.plot(df7['varphi'],df7['chg_consumption'],linestyle='solid')
ax4.plot(df7['varphi'],df7['chg_Qeworld'],linestyle='solid')


ax1.set_title('Extraction',loc='left',fontsize = 8, pad=3)
ax1.set_ylim(-20,20)
ax1.set_yticks([-20,-10 , 0, 10, 20])
ax2.set_title('Production',loc='left',fontsize = 8, pad=3)
ax2.set_ylim(-20,20)
ax2.set_yticks([-20,-10 , 0, 10, 20])
ax3.set_title('Consumption',loc='left',fontsize = 8, pad=3)
ax3.set_ylim(-20,20)
ax2.set_yticks([-20,-10 , 0, 10, 20])
ax4.set_title('Global Emissions',loc='left',fontsize = 8, pad=3)
ax4.set_ylim(-25, 2)
ax4.set_yticks([-20, -15, -10, -5, 0])

ax3.legend(handles = [l1,l2,l3,l4,l5,l6,l7] , labels=['production tax', 'extraction tax',
                                                      'consumption tax','production-consumption',
                                                'extraction-production','extraction-consumption',
                                                'optimal'],loc='upper left', 
           bbox_to_anchor=(-0.1, -0.19),fancybox=True, shadow=False, ncol=4, frameon=False)
plt.savefig('plots/leakage2.pdf', format='pdf')
plt.savefig('plots/leakage2.eps', format='eps')

plt.show()



    