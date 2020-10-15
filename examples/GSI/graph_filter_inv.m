function lambdas = graph_filter_inv(sigma_sq,beta,filter_type)

if sum([sigma_sq(:);beta]<0)>0 % if there are negative values
    error('Error: graph_filter_inv negative variance or beta value!');
end

if isequal(filter_type,'heat')
    
    lambdas = -log(sigma_sq)/beta;
    
elseif isequal(filter_type,'fscale')
    
    lambdas = 1./(beta*sigma_sq);
    lambdas(sigma_sq <= 10^-10) = 0;
    
elseif isequal(filter_type,'fshift')
    
    lambdas = 1./(sigma_sq - beta);
    lambdas(sigma_sq <= 10^-10) = 0;
    
elseif isequal(filter_type,'vshift')
    
    lambdas = 1./(sigma_sq - beta);
    
elseif isequal(filter_type,'hop')
    
    lambdas = 1./(sigma_sq.^(1/beta));
    lambdas(sigma_sq <= 10^-10) = 0;

elseif isequal(filter_type,'norm')
    
   lambdas = zeros(size(sigma_sq));
   lambdas(sigma_sq>1e-2) = 1./(sigma_sq(sigma_sq>1e-2).^2);

elseif isequal(filter_type,'high')
    
    lambdas = sigma_sq./(beta - beta*sigma_sq);
    
else
    error('Error: graph_filter_inv wrong filter_type');
    
end
lambdas(lambdas <= 10^-10) = 0;
end