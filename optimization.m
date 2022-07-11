
function minuswelfare=optimization(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base, varphi, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 

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
%%
minuswelfare=-welfare;

end
           