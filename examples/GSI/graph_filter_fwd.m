function sigma_sq = graph_filter_fwd(lambdas,beta,filter_type)

if sum([lambdas(:);beta]<0)>0 % if there are negative values
    error('Error: graph_filter_fwd negative frequency of beta value!');
end
    
if isequal(filter_type,'heat')
    
    sigma_sq = exp(-beta*lambdas);
    
elseif isequal(filter_type,'fscale')
    
    sigma_sq = 1./(beta*lambdas);

elseif isequal(filter_type,'fshift')
    
     sigma_sq = 1./(beta + lambdas);
     
elseif isequal(filter_type,'vshift')
    
     sigma_sq = (1./(lambdas)) + beta;
     sigma_sq(sigma_sq == Inf) = beta;

elseif isequal(filter_type,'hop')
    
    sigma_sq = 1./(lambdas.^beta);
   
elseif isequal(filter_type,'normal')
    
    sigma_sq = zeros(size(lambdas));
    sigma_sq(lambdas>1e-2) = sqrt(1./lambdas(lambdas>1e-2));
    
elseif isequal(filter_type,'high')
    
    sigma_sq = beta*lambdas./(1+beta*lambdas);
    
else
    error('Error: graph_filter_fwd wrong filter_type');

end

    sigma_sq(sigma_sq <= 10^-10) = 0;
    sigma_sq(sigma_sq == Inf) = 0;

    
end