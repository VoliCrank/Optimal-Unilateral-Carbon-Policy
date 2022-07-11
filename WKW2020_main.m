
% Weisbach, Kortum, Wang (2020) OPTIMAL UNILATERAL CARBON POLICY
% August 26th 2020
% This file is able to calculate: optimals for tax scenarios using hat algebra
% Two methods of calculation: 1. matlab optimization; 2. iteration 

% SEVEN TAX SCENARIOS: 
%  1. BAU; 2. Unilateral Optimal; 3. Pure Extraction Tax; 4. Pure Consumption Tax; 5. Pure Production Tax; 6. Hybrid of Consumption and Extraction Tax; 7. Hybrid of Consumption and Production Tax;
% FIVE REGION SCENARIOS:
%  1. US as home; 2. EU28 as home; 3. OECD37 as home; 4. World as home; 5. China as home 

% Guides to run the program: 
%   step 1. choose one region scenario on line 24
%   step 2. choose one tax scenario on line 77-84 (make sure only single "1" appearing) 
%   step 3. choose calculation method on line 86  (set to 1 if using matlab optimization)

 %% settings
clear
clc;
clf
close all
format long

%some test values
region_scenario = 1;  %in the order of US (1), EU (2), OECD (3), World (4), China (5) as home country
%import BAU data moments
if region_scenario==1  %US as home
Qe=4.4801;
Qestar=27.7959;
CeHH=4.5984;
CeHF=1.1961;
CeFH=0.4216;
CeFF=26.0599;
elseif scenario==2  %EU28 as home
Qe=0.9358;
Qestar=31.3402;
CeHH=2.9506;
CeHF=1.0136;
CeFH=0.5077;
CeFF=27.8042;       
elseif scenario==3  %OECD37 as home
Qe=8.625495; 
Qestar=23.6505;
CeHH=11.29367;
CeHF=2.487537;
CeFH=0.910579;
CeFF=17.58422;  
elseif scenario==4  %World as home
Qe=32.176;
Qestar=0.1;
CeHH=32.146;
CeHF=0.03;
CeFH=0.03;
CeFF=0.07;
elseif scenario==5      %China as home
Qe=7.52274;
Qestar=24.75325;
CeHH=7.345464;
CeHF=0.632472;
CeFH=1.935382;
CeFF=22.36367;  
end

jxbar=CeFH/(CeFH + CeFF);
jmbar=CeHH/(CeHH + CeHF);
Me = CeHH + CeFH;
Mestar = CeFF + CeHF;
Ce = CeHH + CeHF;
Cestar = CeFH + CeFF;
Qeworld = Qe + Qestar;

index=1
for varphi = 0:0.1:3  % marginal damages

%% controls
Baseline = 0;  % only set to 1 if calculate for BAU
Unilateral = 1;  % only set to 1 if calcualte for optimal unilateral policies
pureba = 0;  %only set to 1 if calculate for pure border adjustment
purete = 0;  % only set to 1 if calculate for pure extraction tax
puretc = 0;  % only set to 1 if calculate for pure consumption tax
puretp = 0;  % only set to 1 if calculate for pure production tax
CE_hybrid = 0; % only set to 1 if calculate for hybrid of consumption tax and extraction tax
PC_hybrid = 0; % only set to 1 if calculate for hybrid of production tax and consumption tax
matlab_optimization= 0; % only set to 1 if calculate using matlab own optimization algorithm

%% specific settings for controls above
if Baseline ==1 
    Base = 1;    % determine if home has monopoly pricing power on exports
    Update = 0;  % iteration on/off: 1 means to iterate, 0 means not to update tb
    pureba = 0;
    purete = 0;
    puretc = 0;
    puretp = 0;
    CE_hybrid = 0;
    PC_hybrid = 0;
    matlab_optimization= 0;
end

if Unilateral ==1 
    Base = 0;
    Baseline = 0;
    Update = 1;
    pureba = 0;
    purete = 0;
    puretc = 0;
    puretp = 0;
    CE_hybrid = 0;
    PC_hybrid = 0;
    matlab_optimization= 0;
end

if purete == 1 && matlab_optimization == 0
    Base = 1;
    Update = 1;
    pureba = 0;
    purete = 1;
    puretc = 0;
    puretp = 0;
    CE_hybrid = 0;
    PC_hybrid = 0;
end

if puretc ==1 && matlab_optimization == 0
    Base = 1;
    Update = 1;
    pureba = 0;
    purete = 0;
    puretc = 1;
    puretp = 0;
    CE_hybrid = 0;
    PC_hybrid = 0;
end

if puretp ==1 && matlab_optimization == 0
    warning('matlab_optimization option must be set as 1')
    return
end

if CE_hybrid == 1 && matlab_optimization == 0
    Base = 1;
    Update = 1;
    pureba = 0;
    purete = 0;
    puretc = 0;
    puretp = 0;
    CE_hybrid = 1;
    PC_hybrid = 0;
end

if PC_hybrid == 1 && matlab_optimization == 0
    warning('matlab_optimization option must be set as 1')
    return
end

%% initial values and parameter values
tb = 0; %initial value of border adjustment for iteration
te = 0; %initial value of extraction tax for iteration
tb_mat = [0 1];  %initial value of  border adjustment and proportion of it (prop is mainly used for PC hybrid)
%required parameters (fitted to the BAU)
alpha = 0.85;           % labor share parameter in manufacturing
theta = 4;              % scopevec for comparative advantage
sigma = 0.999;      % elasticity of demand for each individual manufactured good j at Home
sigmastar = 0.999;  % elasticity of demand for each individual manufactured good j at Foreign
epsilonD = alpha + (1 - alpha) * sigma;  %Home's elasticity of demand for embodied energy
epsilonDstar = alpha + (1 - alpha) * sigmastar;  %Foreign's elasticity of demand for embodied energy
epsilonS = 0.5;  %Homes's energy supply elasticity: beta/(1 - beta)
epsilonSstar = 0.5;  %Foreign's energy supply elasticity: betastar/(1 - betastar)
%% matlab optimization
if matlab_optimization==1
    Base=1;
    Update=0;
    
    if purete==1
    tb=0;
    tb_mat=[0 1];
    func=@(te) optimization(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base,  varphi, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 
    options = optimset('TolFun',1e-10,'TolX',1e-10);
    [te, fval] = fminsearch(func,0,options);
    end
    
    if puretc==1
    te=tb;
    func=@(tb_mat) optimization(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base,  varphi, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 
    options = optimset('TolFun',1e-10,'TolX',1e-10);
    X0=[0 1];
    LB=[-inf 1];
    UB=[inf 1];
    [tb_mat,fval] = fmincon(func,X0,[],[],[],[],LB,UB,[],options);
    tb=tb_mat(1)
    prop=tb_mat(2);
    end
    
    if puretp==1
    te=tb;
    func=@(tb_mat)  optimization(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base,  varphi, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 
    options = optimset('TolFun',1e-10,'TolX',1e-10);
    X0=[0 1];
    LB=[0 1];
    UB=[inf 1];
    [tb_mat,fval] = fmincon(func,X0,[],[],[],[],LB,UB,[],options);
    tb=tb_mat(1);
    prop=tb_mat(2);
    end
    
    if CE_hybrid==1
    te=varphi;
    func=@(tb_mat)  optimization(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base,  varphi, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 
    options = optimset('TolFun',1e-10,'TolX',1e-10);
    X0=[0 1];
    LB=[-inf 1];
    UB=[inf 1];
    [tb_mat,fval] = fmincon(func,X0,[],[],[],[],LB,UB,[],options);
    tb=tb_mat(1);
    prop=tb_mat(2);
    end
    
    if PC_hybrid==1
    te=tb;
    func=@(tb_mat) optimization(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base,  varphi, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 
    options = optimset('TolFun',1e-10,'TolX',1e-10);
    X0=[0 0];
    LB=[-inf 0];
    UB=[inf 1];
    [tb_mat,fval] = fmincon(func,X0,[],[],[],[],LB,UB,[],options);
    tb=tb_mat(1);
    prop=tb_mat(2);
    end
end

%% main iteration program

iter = 0;
maxit = 10000;
change = 1;
tol = 1e-10;

while ((iter <= maxit) && (change > tol))
    
%solve for equilibrium
fun = @(pe) equilibrium_hatalgebra(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base,  varphi, pe, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 
pe = fsolve(fun,1)

%% optimal values
jxbar_hat =   (1 - jxbar)^(-1) / (((1 - jxbar)^(-1) - 1) + (1 + (1 - alpha) * tb_mat(1)/pe)^(-theta) * (1 + tb_mat(1)/pe)^((1 - alpha) * theta));
if pureba==1
    jxbar_hat = (1 - jxbar)^(-1) * (1 + tb_mat(1)/pe)^((alpha - 1) * theta)/(((1 - jxbar)^(-1) - 1) * (1 + tb_mat(1)/pe)^((alpha - 1) * theta) + 1);
end

if purete==1
    jxbar_hat = 1;    
end

if puretc ==1
    te=tb_mat(1);
    jxbar_hat = 1;
end

if puretp==1
    te=tb_mat(1);
    jxbar_hat =  (1 - jxbar)^(-1) / ((1 - jxbar)^(-1) - 1 + (1 + tb_mat(1)/pe)^(theta * (1 - alpha)));
    jmbar_hat = (1 + ((1 - jmbar)^(-1) - 1)^(-1)) / (1 + ((1 - jmbar)^(-1) - 1)^(-1) * (1 + tb_mat(1)/pe)^(theta * (1 - alpha)));
    jmbar_prime = jmbar_hat * jmbar;
end

if CE_hybrid==1
    te=varphi;
    jxbar_hat = 1;
end

if PC_hybrid==1
    te=tb_mat(1);
    jxbar_hat =  (1 - jxbar)^(-1) / ((1 - jxbar)^(-1) - 1 + (1 + (tb_mat(1) - tb_mat(2) * tb_mat(1))/pe)^(theta * (1 - alpha)));
    jmbar_hat = (1 + ((1 - jmbar)^(-1) - 1)^(-1)) / (1 + ((1 - jmbar)^(-1) - 1)^(-1) * ((pe + tb_mat(1))/(pe + tb_mat(2) * tb_mat(1)))^(theta * (1 - alpha)));
    jmbar_prime = jmbar_hat * jmbar;
end

jxbar_prime = jxbar_hat * jxbar;

tempFunction = @(i)  (i.^((1 + theta) / theta - 1) ...
           .* (1 - i).^((theta - sigmastar) / theta - 1)) ;
    
Bfunvec_prime = integral(tempFunction,0,jxbar_prime);
    
    %if te is too large, Home stop producing
    petbte = pe + tb_mat(1) - te;
    z = pe + tb_mat(1) >= te;
    petbte = petbte * z;
Qe_hat = (petbte)^epsilonS;
Qe_prime = Qe * Qe_hat;

Qestar_hat = pe^epsilonSstar;
Qestar_prime = Qestar * Qestar_hat;


CeHH_hat = (pe + tb_mat(1))^(-epsilonD);
    if puretp==1|| PC_hybrid==1
        CeHH_hat = (pe + tb_mat(1))^(-epsilonD) * jmbar_hat^(1 + (1 - sigma)/theta);
    end
CeHH_prime = CeHH * CeHH_hat;
   

CeFH_hat = (1 + (1 - sigmastar)/theta) * pe^(-(1 - alpha) * sigmastar) * (pe + tb_mat(1))^(-alpha) * Bfunvec_prime/jxbar^(1+1/theta) * (1 - jxbar)^(sigmastar/theta);
    if Base==1 
        CeFH_hat = pe^(-epsilonDstar) * jxbar_hat^(1 + (1 - sigmastar)/theta);
    end
    if puretp==1
        CeFH_hat = (pe + tb_mat(1))^(-epsilonDstar) * jxbar_hat^(1 + (1 - sigmastar)/theta);
    end
    if PC_hybrid==1
        CeFH_hat = (pe + tb_mat(1) - tb_mat(2) * tb_mat(1))^(-epsilonDstar) * jxbar_hat^(1 + (1 - sigmastar)/theta);
    end
CeFH_prime = CeFH * CeFH_hat;


CeHF_hat = (pe + tb_mat(1))^(-epsilonD);
    if puretp==1
        CeHF_hat = (pe)^(-epsilonD) * ((1 - jmbar_prime)/(1 - jmbar))^(1 + (1 - sigma)/theta);
    end
    if PC_hybrid==1
        CeHF_hat = (pe + tb_mat(2) * tb_mat(1))^(-epsilonD) * ((1 - jmbar_prime)/(1 - jmbar))^(1 + (1 - sigma)/theta);
    end
CeHF_prime = CeHF * CeHF_hat;


CeFF_prime = CeFF * ((1 - jxbar_prime)/(1-jxbar))^(1 + (1 - sigmastar)/theta) * pe^(-epsilonDstar);

%%
VgFH = CeFH /(1 - alpha);
VgFH_prime = VgFH * pe^((1 - sigmastar) * (1 - alpha)) * (1 - (1 - jxbar_prime)^(1 + (1 - sigmastar)/theta))/ (jxbar * (1 - jxbar)^( (1-sigmastar)/theta));
    if Base==1 
        VgFH_hat = pe * CeFH_hat;
        VgFH_prime = VgFH * VgFH_hat;
    end
    if puretp==1
        VgFH_hat = (pe + tb_mat(1)) * CeFH_hat;
        VgFH_prime = VgFH * VgFH_hat;
    end
    if PC_hybrid==1
        VgFH_hat = (pe + tb_mat(1) - tb_mat(2) * tb_mat(1)) * CeFH_hat;
        VgFH_prime = VgFH * VgFH_hat;
    end


VgHF = CeHF/(1 - alpha);
VgHF_hat = (pe + alpha * tb_mat(1)) * CeHF_hat;
    if puretp==1
        VgHF_hat = pe * CeHF_hat;
    end
    if PC_hybrid==1
        VgHF_hat = (pe + alpha * tb_mat(2) * tb_mat(1)) * CeHF_hat;
    end
VgHF_prime = VgHF * VgHF_hat;


Me = CeHH + CeFH;
Me_prime = CeHH_prime + CeFH_prime;
Me_hat = Me_prime/Me;
Mestar = CeFF + CeHF;
Mestar_prime = CeFF_prime + CeHF_prime;
Ce = CeHH + CeHF;
Ce_prime = CeHH_prime + CeHF_prime;
Ce_hat = Ce_prime/Ce;
Cestar = CeFF + CeFH;
Cestar_prime = CeFF_prime + CeFH_prime;
Xe = Qe - Me;
Xe_prime = Qe_prime - Me_prime;
Xg = VgFH - VgHF;
Xg_prime = VgFH_prime - VgHF_prime;
Xestar_prime = Qestar_prime - Mestar_prime;
Qeworld = Qe + Qestar;
Qeworld_prime=Qe_prime+Qestar_prime;
Meworld_prime=CeHH_prime+CeFH_prime+CeHF_prime+CeFF_prime;

Vg = Ce /(1-alpha);
Vg_prime_hat = (pe + tb_mat(1)) * Ce_hat;
Vg_prime = Vg * Vg_prime_hat;
    if puretp==1
        Vg_prime = CeHH_prime/(1-alpha) * (pe + tb_mat(1)) + CeHF_prime/(1-alpha) * pe; 
    end
    if PC_hybrid==1
        Vg_prime = CeHH_prime/(1-alpha) * (pe + tb_mat(1)) + CeHF_prime/(1-alpha) * (pe + tb_mat(2) * tb_mat(1)); 
    end

    
Lg = alpha/(1-alpha) * Me;
Lg_prime = alpha/(1-alpha) * (pe + tb_mat(1)) * Me_prime;
    if puretc==1 || CE_hybrid==1
        Lg_prime = alpha/(1-alpha) * (pe + tb_mat(1)) * CeHH_prime + alpha * VgFH_prime;
    end
    if PC_hybrid==1
        Lg_prime = alpha/(1-alpha) * (pe + tb_mat(1)) * CeHH_prime + alpha/(1-alpha) * (pe + tb_mat(1) - tb_mat(2) * tb_mat(1)) * CeFH_prime;
    end

leakage1 = -(Qestar_prime - Qestar)/(Qeworld_prime - Qeworld);
leakage2 = -(Mestar_prime - Mestar)/(Qeworld_prime - Qeworld);
leakage3 = -(Cestar_prime - Cestar)/(Qeworld_prime - Qeworld);

delta_U = -(epsilonS/(epsilonS + 1)) * Qe * (petbte^(epsilonS + 1) - 1) - (Lg_prime - Lg) + (Xg_prime - Xg) ...
           + (pe * Xe_prime - Xe) + sigma/(sigma-1) * (Vg_prime - Vg) ...
           - varphi * (Qeworld_prime - Qeworld);
       
welfare = delta_U/Vg;             
%% update taxes

%BAU
if Baseline==1
    oldtb = tb_mat(1);
    change = abs(tb-oldtb);
end

%optimal unilateral case: both extraction tax and border adjustment
if Unilateral==1
    optimaltb = varphi * (epsilonSstar * Qestar_prime)/(epsilonSstar * Qestar_prime + epsilonDstar * CeFF_prime) ...
                 +(pe * Xestar_prime)/(epsilonSstar*Qestar_prime+epsilonDstar*CeFF_prime) ...
                 +(pe * CeHF_prime - (pe + tb_mat(1)) * sigmastar * CeFH_prime - (1 - epsilonDstar) * VgFH_prime)/ (epsilonSstar * Qestar_prime + epsilonDstar * CeFF_prime);
    oldtb = tb_mat(1);
    if Update ==1
    tb_mat(1) = tb_mat(1) + 0.2*(optimaltb-tb_mat(1));
    te = varphi;
    end
    change = abs(tb_mat(1)-oldtb);
end


%pure extraction tax case
if purete==1
    optimalte= varphi * (epsilonD * Ce_prime + epsilonDstar * Cestar_prime)/(epsilonD * Ce_prime + epsilonDstar * Cestar_prime + epsilonSstar * Qestar_prime) ...
           +(pe * (Qe_prime - Ce_prime))/(epsilonD * Ce_prime + epsilonDstar * Cestar_prime + epsilonSstar * Qestar_prime)
    oldte = te;

    if Update ==1
    te = optimalte;
    tb_mat(1) = 0;
    end
    change = abs(te-oldte)
end

%pure consumption tax case
if puretc==1
    optimaltb = varphi * (epsilonSstar * Qestar_prime + epsilonS * Qe_prime)/(epsilonSstar * Qestar_prime + epsilonS * Qe_prime + epsilonDstar * Cestar_prime) ...
                 +(pe * (Qestar_prime - Cestar_prime))/(epsilonSstar * Qestar_prime + epsilonS * Qe_prime + epsilonDstar * Cestar_prime);
    oldtb = tb_mat(1);
    if Update ==1
    tb_mat(1) = optimaltb;
    te = tb_mat(1);
    end
    change = abs(tb_mat(1)-oldtb);
end

%pure production tax case
if puretp==1
    oldtb = tb_mat(1);
    change = abs(tb_mat(1)-oldtb);
end

%CE_hybrid tax case
if CE_hybrid==1
    optimaltb = varphi * epsilonSstar * Qestar_prime / (epsilonSstar * Qestar_prime + epsilonDstar * Cestar_prime) ...
                 -(pe * (Qe_prime - Ce_prime))/(epsilonSstar * Qestar_prime + epsilonDstar * Cestar_prime)
    oldtb = tb_mat(1);
    if Update ==1
    tb_mat(1) = tb_mat(1)+0.9*(optimaltb-tb_mat(1));
    te = varphi;
    end
    change = abs(tb_mat(1)-oldtb);
end

%PC_hybrid tax case
if PC_hybrid==1
    oldtb = tb_mat(1);
    change = abs(tb_mat(1)-oldtb);
end

varphi
iter=iter+1

if iter>=maxit
exit from loop % *not converging*
end %exit flag

end

tb=tb_mat(1);
prop=tb_mat(2);  %proportion of tb (used for the PC-hybrid case): 1--same as in the consumption tax; 0--same as in the pure production tax
prop_vec(index)=prop;
varphi_vec(index)=varphi;
pe_vec(index)=pe;
tb_vec(index)=tb;
te_vec(index)=te;
Me_vec(index)=Me_prime;
Mestar_vec(index)=Mestar_prime;
Ce_vec(index)=Ce_prime;
Cestar_vec(index)=Cestar_prime;
Qe_vec(index)=Qe_prime;
Qestar_vec(index)=Qestar_prime;
Qeworld_vec(index)=Qeworld_prime;
welfare_vec(index)=welfare;
leakage1_vec(index)=leakage1;
leakage2_vec(index)=leakage2;
chg_consumption(index)=leakage3;
Vg_vec(index)=Vg_prime;
te_rate_vec(index)=te/pe;
tb_rate_vec(index)=tb/pe;
change(index)=change;

index=index+1
end

%% save the results
% baseline
if Baseline==1
filename=['baselines_sce' num2str(region_scenario) '.mat']
save(['matfiles/' filename])
end

% optimal
if Unilateral==1
filename=['optimals_sce' num2str(region_scenario) '.mat']
save(['matfiles/' filename])
end


% pure tp
if puretp==1
filename=['puretp_sce' num2str(region_scenario) '.mat']
save(['matfiles/' filename])
end


% pure tc
if puretc==1
filename=['puretc_sce' num2str(region_scenario) '.mat']
save(['matfiles/' filename])
end


% pure te
if purete==1
filename=['purete_sce' num2str(region_scenario) '.mat']
save(['matfiles/' filename])
end


% CE_hybrid
if CE_hybrid==1
filename=['CE_hybrid_sce' num2str(region_scenario) '.mat']
save(['matfiles/' filename])
end

% PC_hybrid
if PC_hybrid==1
filename=['PC_hybrid_sce' num2str(region_scenario) '.mat']
save(['matfiles/' filename])
end

   