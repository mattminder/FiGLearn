clear all
clc
close all

%% Simulation parameters
my_eps_outer = 10^(-4); my_eps_inner = 10^(-6);
max_iterations = 20;

%% Generate model

BS = 8;
filter_list = {'heat','high','normal'};
for filter_type_cell = filter_list
    filter_type = string(filter_type_cell);
    disp(filter_type);
   
    for k=3:2:7 
%         load(['As_' num2str(k*10) '_diffkernel.mat']);
%         As_diffkernel_res = zeros(size(As_diffkernel));
%         As_diffkernel_true = zeros(size(As_diffkernel));
        % load sample data
%         load(['samples_' num2str(k*10) '_heat.mat']);
        load(['Ls_' num2str(k*10) '.mat']);
        load(['As_' num2str(k*10) '.mat']);
        % init results matrix
        As_GSI_est = zeros(size(Ls));
        Ls_GSI_est = zeros(size(Ls));
        % iterate over all graphs in each category (i.e., 30, 50, 70)
        for i=1:size(Ls,1)
%             A_connectivity = squeeze(As(1,:,:));  
%             A_mask = A_connectivity;
%             [Ltrue,~,~] = generateRandomGraphFromConnectivity(A_connectivity,3,0,'uniform');
              num_vertices = size(Ls,2);
              Ltrue = Ls;


            % generate filter
            %filter_type = 'hop'; beta = 3; % beta-hop filter with true beta parameter is set here
           
            % initializaions
            if strcmp(filter_type,'heat')
                beta = 0.1;
            elseif  strcmp(filter_type,'high')
                beta = 0.5;
                
            else 
                beta = 0.5;
 
            end 
            graph_filter_ideal = @(x)(graph_filter_fwd(x,beta,filter_type) );

            % generate graph system
%             [h_L,h_L_sqrMatrix] = generateFilteredModel(Ltrue,graph_filter_ideal);


            %% Generate data samples based on h_L
%             S_data_cell = generateRandomDataFromSqrtM(h_L_sqrMatrix,30); % creates data with 30 samples per vertex
%             S_data = S_data_cell{1}; S_data = 0.5*(S_data + S_data');
            
            samples_sq = squeeze(double(samples_heat(i,:,:)));  
            % sample covariance
            S_data = cov(samples_sq,1);

            %% Algorithm implementation 
            beta_current = 1; % initialize beta
            % Eigendecompose data
            [U,sigma_sq_C] = createBasis(S_data,'descend');
            sigma_sq_C(sigma_sq_C <= 10^-10) = 0;

            for repeat=1:max_iterations
             disp(['-- Iteration ' num2str(repeat) '--']);

            % Step I: Prefiltering step
            lambdas_current = graph_filter_inv(sigma_sq_C,beta_current,filter_type);
            current_sigmas = 1./lambdas_current; 
            current_sigmas(current_sigmas==Inf)=0;

            % construct unfiltered S
            S_prefiltered = U * diag(abs(current_sigmas)) * U'; 
            S_prefiltered = 0.5*(S_prefiltered + S_prefiltered'); % symmetrize (in case of numerical inaccuracies)
            max_eig_of_S_data = max(current_sigmas);
            S_prefiltered = S_prefiltered/max_eig_of_S_data; % normalize

            % Step II: Graph learning step
            Laplacian = estimate_cgl(S_prefiltered,ones(num_vertices),eps,my_eps_outer,my_eps_inner,max_iterations);
            Laplacian = Laplacian/max_eig_of_S_data; % normalize

            % Step III: Parameter estimation step
            estimated_lambdas = diag(U' * Laplacian * U); estimated_lambdas(estimated_lambdas <= 10^-10) = 0;
            estimated_sigmas = graph_filter_fwd(estimated_lambdas,repeat,filter_type);
            h_L_current = U * diag(estimated_sigmas) * U'; h_L_current = 0.5*(h_L_current + h_L_current');
            error_est(repeat) = norm(S_data-h_L_current,'fro')/norm(S_data,'fro');
            % convergence criterion
            if repeat > 1 && abs(error_est(repeat-1) - error_est(repeat)) > 0.2
                 disp('*** Algorithm has converged ***');
                 break;
            end
            beta_current = beta_current + 1;
            end


            disp(['Estimated beta = ' num2str(beta_current) ]);
            disp(' Figure 1 shows the ground truth graph');
            disp(' Figure 2 shows the estimated graph');

            %f1=figure(1);
%             As_diffkernel_true(i,:,:) = laplacianToAdjacency(Ltrue,eps);
            %draw_grid_graph(laplacianToAdjacency(Ltrue,eps),BS);
            %axis square
            %movegui(f1,'west');

            %f2=figure(2);
            As_GSI_est(i,:,:) = laplacianToAdjacency(Laplacian,0.000);
            Ls_GSI_est(i,:,:) = Laplacian;
            %draw_grid_graph(laplacianToAdjacency(Laplacian,eps),BS);
            %axis square
            %movegui(f2,'east');
   
        end
    % save resutls
        if strcmp(filter_type,'heat')
            save(['mat_files/As_' num2str(k*10) '_' 'heatkernel_GSI_res.mat'], 'As_GSI_est');
            save(['mat_files/Ls_' num2str(k*10) '_' 'heatkernel_GSI_res.mat'], 'Ls_GSI_est');

        elseif strcmp(filter_type,'high')
            save(['mat_files/As_' num2str(k*10) '_' 'highkernel_GSI_res.mat'], 'As_GSI_est');
            save(['mat_files/Ls_' num2str(k*10) '_' 'highkernel_GSI_res.mat'], 'Ls_GSI_est');

        elseif strcmp(filter_type,'normal')
            save(['mat_files/As_' num2str(k*10) '_' 'normalkernel_GSI_res.mat'], 'As_GSI_est');
            save(['mat_files/Ls_' num2str(k*10) '_' 'normalkernel_GSI_res.mat'], 'Ls_GSI_est');
        else
            error('Error: graph_filter_fwd wrong filter_type');
        end
    end
end
    