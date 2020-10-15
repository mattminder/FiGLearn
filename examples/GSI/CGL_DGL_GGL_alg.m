clear 
clc
close all


my_eps_outer = 1e-4; my_eps_inner = 1e-6; max_cycles = 40;
scale = 1;
isNormalized = 0;


% Filters 
filter_keySet = {'heat','norm','high'};
valueSet = [1 2 3];
M = containers.Map(filter_keySet,valueSet);
for filter=1:length(M)
    k = keys(M);
    filter_type = k{filter};
    disp(filter_type);
    for k=3:2:7 
        % load sample data
        load(['samples_' num2str(k*10) '_' filter_type '.mat']);
        load(['Ls_' num2str(k*10) '.mat']);
        
        % init results matrix
        As_GGL_est = zeros(size(Ls));
        Ls_GGL_est = zeros(size(Ls));
        
        As_DDGL_est = zeros(size(Ls));
        Ls_DDGL_est = zeros(size(Ls));
        
        As_CGL_est = zeros(size(Ls));
        Ls_CGL_est = zeros(size(Ls));

        % iterate over 20 graphs in each category (i.e., 30, 50, 70)
        for i=1:size(Ls,1)
            samples_sq = squeeze(double(samples(i,:,:)));  

            % sample covariance
            S = cov(samples_sq,1);

            % initializaions
            if strcmp(filter_type,'heat')
                beta = 0.1;
            elseif  strcmp(filter_type,'high')
                beta = 0.5;
                
            else 
                beta = 0.5;
 
            end 
        % for binary data we add +1/3 to diagonals(suggested by Banerjee et al. ''Model Selection Through Sparse Maximum Likelihood Estimation for Multivariate Gaussian or Binary Data (2008)
        S = S  + (1/3)*eye(size(S));
        A_mask=ones(size(S)) - eye(size(S));
        alpha = 0.00;
        [Laplacian_ggl,~,convergence] = estimate_ggl(S,A_mask,alpha,my_eps_outer,my_eps_inner,max_cycles,2);
        [Laplacian_ddgl,~,convergence] = estimate_ddgl(S,A_mask,alpha,my_eps_outer,my_eps_inner,max_cycles,2);
        [Laplacian_cgl,~,convergencen] = estimate_cgl(S,A_mask,alpha,my_eps_outer,my_eps_inner,max_cycles,2);

        Laplacian_ggl(abs(Laplacian_ggl) < my_eps_outer) = 0;  % threshold 
        Laplacian_ddgl(abs(Laplacian_ddgl) < my_eps_outer) = 0;  % threshold 
        Laplacian_cgl(abs(Laplacian_cgl) < my_eps_outer) = 0;  % threshold 

        As_GGL_est(i,:,:) = laplacianToAdjacency(Laplacian_ggl,0.000);
        Ls_GGL_est(i,:,:) = Laplacian_ggl;
        
        As_DDGL_est(i,:,:) = laplacianToAdjacency(Laplacian_ddgl,0.000);
        Ls_DDGL_est(i,:,:) = Laplacian_ddgl;
        
        As_CGL_est(i,:,:) = laplacianToAdjacency(Laplacian_cgl,0.000);
        Ls_CGL_est(i,:,:) = Laplacian_cgl;
        end 
        
        % save resutls
        save(['mat_files/As_' num2str(k*10) '_' filter_type 'kernel_GGL_res.mat'], 'As_GGL_est');
        save(['mat_files/Ls_' num2str(k*10) '_' filter_type 'kernel_GGL_res.mat'], 'Ls_GGL_est');
        
        save(['mat_files/As_' num2str(k*10) '_' filter_type 'kernel_DDGL_res.mat'], 'As_DDGL_est');
        save(['mat_files/Ls_' num2str(k*10) '_' filter_type 'kernel_DDGL_res.mat'], 'Ls_DDGL_est');
        
        save(['mat_files/As_' num2str(k*10) '_' filter_type 'kernel_CGL_res.mat'], 'As_CGL_est');
        save(['mat_files/Ls_' num2str(k*10) '_' filter_type 'kernel_CGL_res.mat'], 'Ls_CGL_est');
    end
end