clear all
close all

%% Simulation parameters
my_eps_outer = 10^(-4); my_eps_inner = 10^(-6);
max_iterations = 20;

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
        As_GSI_est = zeros(size(Ls));
        Ls_GSI_est = zeros(size(Ls));

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
            
            graph_filter_ideal = @(x)(graph_filter_fwd(x,beta,filter_type) );
            [U,sigma_sq_C] = createBasis(S,'descend');
            max_sigma=(max(sigma_sq_C));
            sigma_orig = sigma_sq_C/max_sigma;
            
            %% Algorithm implementation 
            beta_current = 1; % initialize beta
            for repeat=1:max_iterations
            disp(['-- Iteration ' num2str(repeat) '--']);

                % step I: prefilter
                sigma_sq_C = sigma_sq_C/max_sigma; sigma_sq_C(sigma_sq_C <= 10^-10) = 0;
                lambdas_current = graph_filter_inv(sigma_sq_C,beta,filter_type);
                orig_sigmas = 1./lambdas_current; orig_sigmas(orig_sigmas==Inf)=0;
                S_prefiltered = U * diag(orig_sigmas) * U';

                % step II: graph learning 
                Laplacian = estimate_cgl(S_prefiltered,ones(size(S_prefiltered)),0.000,10^-5,10^-7,40);

                % step III: filter parameter estimation (for a desired filter type a filter parameter selection step)
                %  Note: for exponential filter filter parameter selection step can be skipped, 
                %        becayse the output graphs are scaled versions of eachother for different beta parameter
                %        please refer to the paper for further details
                estimated_lambdas = diag(U' * Laplacian * U); estimated_lambdas(estimated_lambdas <= 10^-10) = 0;
                estimated_sigmas = graph_filter_fwd(estimated_lambdas,repeat,filter_type);
                h_L_current = U * diag(estimated_sigmas) * U'; h_L_current = 0.5*(h_L_current + h_L_current');
                error_est(repeat) = norm(S-h_L_current,'fro')/norm(S,'fro');
                % convergence criterion
                if repeat > 1 && abs(error_est(repeat-1) - error_est(repeat)) > 0.2
                     disp('*** Algorithm has converged ***');
                     break;
                end
                beta_current = beta_current + 1;
            end 
            
            disp(['Estimated beta = ' num2str(beta_current) ]);
            % show resulting graph on the US map
            % draw_us_temp_graph(Laplacian, center_vector);
            As_GSI_est(i,:,:) = laplacianToAdjacency(Laplacian,0.000);
            Ls_GSI_est(i,:,:) = Laplacian;
        end 
        
        % save resutls
        save(['mat_files/As_' num2str(k*10) '_' filter_type 'kernel_GSI_res.mat'], 'As_GSI_est');
        save(['mat_files/Ls_' num2str(k*10) '_' filter_type 'kernel_GSI_res.mat'], 'Ls_GSI_est');  
    end
end


