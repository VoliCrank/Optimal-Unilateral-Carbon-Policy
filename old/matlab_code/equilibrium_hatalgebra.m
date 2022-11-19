function diff=equilibrium_hatalgebra(tb_mat, pureba, purete, puretp, puretc, CE_hybrid, PC_hybrid, Base, varphi, pe, te, jxbar, jmbar, alpha, theta, sigma, sigmastar, Qe, Qestar, CeHH, CeFH, CeFF, CeHF, epsilonD, epsilonDstar, epsilonS, epsilonSstar ) 


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
diff = Qe_prime + Qestar_prime - (CeHH_prime + CeFH_prime + CeHF_prime + CeFF_prime);

end