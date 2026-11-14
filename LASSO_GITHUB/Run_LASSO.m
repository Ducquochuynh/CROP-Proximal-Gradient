%clc;
rng(0, 'twister');   % Fix random seed for reproducibility

% Problem setup
m = 6000;           % number of rows
n = 3000;           % number of columns
cond_number = 1e8;  % desired condition number
sparsity = 0.7;     % fraction of nonzero entries in x_true
noise_level = 0.5;  % noise level
lambda_frac = 1e-2; % fraction of lambda_max

% Generate ill-conditioned matrix A
[U, ~] = qr(randn(m,m),0);   % m x m
[V, ~] = qr(randn(n,n),0);   % n x n

k = min(m,n);                 % rank
s = logspace(0, -log10(cond_number), k); % singular values
S = diag(s);

if m >= n
    A = U(:,1:n) * S * V;     % m x n
else
    A = U * S * V(1:m,:);     % m x n
end

% Scale columns differently → numeric instability
A = A .* repmat(logspace(2,-2,n), m, 1);


% Generate sparse true solution x_true
x_true = sprandn(n,1,sparsity);

% Generate observation vector b with noise
b = A*x_true + noise_level*randn(m,1);

% Set Lasso regularization parameter
lambda_max = norm(A' * b, Inf);
lambda = lambda_frac * lambda_max;

%======================
x0 = zeros(n,1);
L = norm(A)^2;
gamma = 1 / L;
delta = 1e-4;
maxit = 2000;
tol = 1e-13;


%% Display LASSO problem information
[m, n] = size(A);
cond_A = cond(A);

fprintf('\n========== LASSO PROBLEM INFORMATION ==========\n');
fprintf('Size of A: %d x %d\n', m, n);
fprintf('Target condition number: %.6e\n', cond_number);
fprintf('Actual condition number of A: %.6e\n', cond_A);
fprintf('Sparsity of x_true: %.2f%%\n', 100*nnz(x_true)/n);
fprintf('Number of nonzeros in x_true: %d\n', nnz(x_true));
fprintf('Noise level: %.6e\n', noise_level);
fprintf('Lambda fraction: %.6e\n', lambda_frac);
fprintf('Lambda max: %.12e\n', lambda_max);
fprintf('Lambda used: %.12e\n', lambda);
fprintf('Lipschitz constant L: %.12e\n', L);
fprintf('Stepsize gamma: %.12e\n', gamma);
fprintf('Maximum iterations: %d\n', maxit);
fprintf('Stopping tolerance: %.2e\n', tol);
fprintf('================================================\n\n');


%======================= RUNNING TEST================================
% Independent timing: works even when solver info lacks cpu_time.
solver_cpu = zeros(10,1);
solver_wall = zeros(10,1);
% Run PG (Vannila Proximal Gradient Algorithm)
fprintf('Running PGA...\n');
timer_wall = tic; timer_cpu = cputime;
[x_pga, info_pga] = pg_lasso(A, b, lambda, x0, gamma, maxit, tol);
solver_wall(1) = toc(timer_wall);
solver_cpu(1) = cputime - timer_cpu;
%=======================
% Run Accelerated Proximal Gradient (FISTA)
fprintf('Running FISTA...\n')
timer_wall = tic; timer_cpu = cputime;
[x_fis, info_fis] = fista_lasso(A, b, lambda, x0, gamma, maxit, tol);
solver_wall(2) = toc(timer_wall);
solver_cpu(2) = cputime - timer_cpu;
%=========================
% Run AA-PG (Anderson Acceleration Proximal Gradient Algorithm)
fprintf('Running AA-PG...\n');
% m = 10
fprintf('Running m = 10...\n');
timer_wall = tic; timer_cpu = cputime;
[x_aa10, info_aa10, acc_aa10] = aa_pg_lasso(A, b, lambda, x0, gamma, 10, maxit, tol);
solver_wall(3) = toc(timer_wall);
solver_cpu(3) = cputime - timer_cpu;
% m = 20
fprintf('Running m = 20...\n');
timer_wall = tic; timer_cpu = cputime;
[x_aa20, info_aa20, acc_aa20] = aa_pg_lasso(A, b, lambda, x0, gamma, 20, maxit, tol);
solver_wall(4) = toc(timer_wall);
solver_cpu(4) = cputime - timer_cpu;


%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run CROP-PG (CROP Proximal Gradient Algorithm)
fprintf('Running CROP-PG...\n');
% m = 10
fprintf('Running m = 10...\n');
timer_wall = tic; timer_cpu = cputime;
[x_crop10, info_crop10, acc_crop10] = crop_pg_lasso(A, b, lambda, x0, gamma, delta, 10, maxit, tol,1000);
solver_wall(5) = toc(timer_wall);
solver_cpu(5) = cputime - timer_cpu;
% m = 20
fprintf('Running m = 20...\n');
timer_wall = tic; timer_cpu = cputime;
[x_crop20, info_crop20, acc_crop20] = crop_pg_lasso(A, b, lambda, x0, gamma, delta, 20, maxit, tol,1000);
solver_wall(6) = toc(timer_wall);
solver_cpu(6) = cputime - timer_cpu;
%=============================================
fprintf('Running CROP-PG-Exact...\n');
% m = 10
fprintf('Running m = 10...\n');
timer_wall = tic; timer_cpu = cputime;
[x_crop_exact10, info_crop_exact10, acc_crop_exact10] = crop_pg_lasso_exact(A, b, lambda, x0, gamma, delta, 10, maxit, tol,1000);
solver_wall(7) = toc(timer_wall);
solver_cpu(7) = cputime - timer_cpu;
% m = 20
fprintf('Running m = 20...\n');
timer_wall = tic; timer_cpu = cputime;
[x_crop_exact20, info_crop_exact20, acc_crop_exact20] = crop_pg_lasso_exact(A, b, lambda, x0, gamma, delta, 20, maxit, tol,1000);
solver_wall(8) = toc(timer_wall);
solver_cpu(8) = cputime - timer_cpu;

%=============================================
% Run adaptive proximal gradient (AdaPG^{q,q/2})
q_adapg = 1.5;  % choose any q in [1,2]
fprintf('Running AdaPG (q = %.2f)...\n', q_adapg);
timer_wall = tic; timer_cpu = cputime;
[x_adapg, info_adapg] = adapg_lasso(A, b, lambda, x0, gamma, maxit, tol, q_adapg);
solver_wall(9) = toc(timer_wall);
solver_cpu(9) = cputime - timer_cpu;
%=============================================
%=============================================

% Run PGM-pc (Algorithm 5.1, known L = norm(A)^2)
fprintf('Running PGM-pc...\n');
L_pgm = 1/gamma; % same smoothness constant as the PG/FISTA baseline
timer_wall = tic; timer_cpu = cputime;
[x_pgmpc, info_pgmpc] = pgmpc_lasso(A,b,lambda,x0,gamma,maxit,tol,L_pgm);
solver_wall(10) = toc(timer_wall);
solver_cpu(10) = cputime - timer_cpu;
%=============================================


solutions={x_pga,x_fis,x_aa10,x_aa20,x_crop10,x_crop20, ...
    x_crop_exact10,x_crop_exact20,x_adapg,x_pgmpc};
info={info_pga,info_fis,info_aa10,info_aa20,info_crop10,info_crop20, ...
    info_crop_exact10,info_crop_exact20,info_adapg,info_pgmpc};
accepted=[NaN;NaN;acc_aa10;acc_aa20;acc_crop10;acc_crop20; ...
    acc_crop_exact10;acc_crop_exact20;NaN;NaN];


% ================================================================
% COMMON POST-PROCESSING (identical across LASSO, L1-RLR and NRR)
% All 10 solvers and their original stopping rules are unchanged.
% ================================================================
method_names_plot = {'PG','FISTA','AA-PG (m = 10)','AA-PG (m = 20)', ...
    'CROP-PG-Inexact (m = 10)','CROP-PG-Inexact (m = 20)', ...
    'CROP-PG-Exact (m = 10)','CROP-PG-Exact (m = 20)','AdaPG','PGM-pc'};
Method = string(method_names_plot(:));
nmethods = numel(Method);
Iterations = zeros(nmethods,1);
CPUTime_s = zeros(nmethods,1);
WallTime_s = zeros(nmethods,1);
FinalObjective = zeros(nmethods,1);
FinalRelChange = nan(nmethods,1);
PGResidual = nan(nmethods,1);
AcceptedPct = nan(nmethods,1);
StopMet = false(nmethods,1);

% Crucially, gamma_eval is FIXED for ALL methods, including AdaPG and
% PGM-pc. For NRR, L includes the ridge parameter.
gamma_eval = gamma;
for j = 1:nmethods
    rec = info{j};
    xfinal = solutions{j}(:);
    Iterations(j) = rec.k;
    CPUTime_s(j) = solver_cpu(j);
    WallTime_s(j) = solver_wall(j);
    FinalObjective(j) = rec.f_hist(end);
    if ~isempty(rec.res_hist)
        FinalRelChange(j) = rec.res_hist(end);
    end
    if isfield(rec,'stop_met')
        StopMet(j) = logical(rec.stop_met);
    else
        StopMet(j) = isfinite(FinalRelChange(j)) && FinalRelChange(j) < tol;
    end
    if j >= 3 && j <= 8
        AcceptedPct(j) = accepted(j);
    end
    grad_x = A'*(A*xfinal-b);
    prox_x = sign(xfinal-gamma_eval*grad_x) .* ...
        max(abs(xfinal-gamma_eval*grad_x)-gamma_eval*lambda,0);
    PGResidual(j) = norm((xfinal - prox_x)/gamma_eval) ...
                    / max(1,norm(xfinal));
end

summary = table(Method,Iterations,CPUTime_s,WallTime_s,FinalObjective, ...
    FinalRelChange,PGResidual,AcceptedPct,StopMet);
disp(summary);
fprintf('\nFinal normalized PG residuals (common gamma = %.12e):\n',gamma_eval);
disp(summary(:,{'Method','Iterations','CPUTime_s','FinalObjective', ...
    'PGResidual','StopMet'}));
[phi_ref, ref_idx] = min(FinalObjective);
fprintf('Common best FINAL objective: %.16e (%s)\n',phi_ref,Method(ref_idx));

outdir = fullfile(pwd,'results_LASSO_PGResidual');
if ~exist(outdir,'dir'), mkdir(outdir); end
writetable(summary,fullfile(outdir,'LASSO_10Methods_summary.csv'));

save(fullfile(outdir,'LASSO_10Methods_results.mat'), ...
    'summary','info','solutions','phi_ref','gamma_eval', ...
    'lambda','lambda_frac','lambda_max','L','gamma', ...
    'cond_A','cond_number','m','n','sparsity','noise_level', ...
    'maxit','tol','method_names_plot','-v7.3');


% Three standardized figures, logarithmic Y axes throughout.
% Objective gap: reference is best observed FINAL value, NOT a certified optimum.
plot_pg_comparison(info,method_names_plot,phi_ref,outdir,'LASSO');

% ================================================================
% Local plotting function: matched styles, 800x500, log Y, sparse markers
% ================================================================
function plot_pg_comparison(infos,labels,phi_ref,outdir,tag)
    colors = lines(numel(infos));
    styles = {'-','--','-','--','-','-','-.','-.','--','-'};
    markers = {'o','s','d','>','x','p','^','p','h','v'};
    gap_floor = 1e-12 * max(1,abs(phi_ref));
    residual_floor = 1e-16;
    for kind=1:3
        fig=figure('Name',tag,'Color','w','Position',[200 200 800 500]);
        hold on;
        for j=1:numel(infos)
            rec=infos{j};
            if kind==2
                y=rec.res_hist(:);
                k=rec.k;
                % Align iteration count to the actual number of samples.
                x=(k-numel(y)+1:k)';
                y(~isfinite(y) | y<=residual_floor)=NaN;
            else
                y=rec.f_hist(:)-phi_ref;
                y(~isfinite(y) | y<=gap_floor)=NaN;
                if kind==1
                    % Some solvers record x0 as the first objective sample.
                    if numel(y)==rec.k+1
                        x=(0:rec.k)';
                    else
                        x=(rec.k-numel(y)+1:rec.k)';
                    end
                else
                    x=rec.t_hist(:);  % wall-clock time used in histories
                    if isfield(rec,'setup_time') && ~isempty(rec.setup_time)
                        x=x+rec.setup_time;
                    end
                    if numel(x)==numel(y)-1
                        x=[0;x];  % only for histories that include initial objective
                    elseif numel(x)~=numel(y)
                        error('Time/objective length mismatch for %s.',labels{j});
                    end
                end
            end
            assert(numel(x)==numel(y),'X/Y mismatch for %s.',labels{j});
            marker_step=max(1,ceil(numel(y)/20));
            semilogy(x,y,'Color',colors(j,:), ...
                'LineStyle',styles{j},'Marker',markers{j}, ...
                'MarkerIndices',1:marker_step:numel(y), ...
                'MarkerSize',5,'LineWidth',2.0,'DisplayName',labels{j});
        end
        grid on;
        set(gca,'YScale','log','FontSize',16,'LineWidth',1);
        if kind==1
            xlabel('Iteration Number','Interpreter','latex','FontSize',18);
            ylabel('$\phi(x_k)-\phi_{\mathrm{ref}}$', ...
                'Interpreter','latex','FontSize',18);
            title('Objective vs Iteration','FontSize',17);
            fname='Objective_vs_Iteration';
        elseif kind==2
            xlabel('Iteration Number','Interpreter','latex','FontSize',18);
            ylabel('$|\phi(x_{k+1})-\phi(x_k)|/(1+|\phi(x_k)|)$', ...
                'Interpreter','latex','FontSize',17);
            title('Residual vs Iteration','FontSize',17);
            fname='Residual_vs_Iteration';
        else
            xlabel('CPU time (s)','Interpreter','latex','FontSize',18);
            ylabel('$\phi(x_k)-\phi_{\mathrm{ref}}$', ...
                'Interpreter','latex','FontSize',18);
            title('Objective vs Time','FontSize',17);
            fname='Objective_vs_Time';
        end
        legend('show','Location','northeast','FontSize',10,'Box','off');
        exportgraphics(fig,fullfile(outdir,[tag '_' fname '.pdf']), ...
            'ContentType','vector');
        exportgraphics(fig,fullfile(outdir,[tag '_' fname '.png']), ...
            'Resolution',200);
    end
end
