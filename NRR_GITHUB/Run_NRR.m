%% Ten-method NRR comparison (optimized)
% Keep this script and nrr_operators.m in the same folder as the six solvers.
clear; clc; 
rng(0,'twister');
% Adjust problem size as needed; original experiment: rows=5000, cols=2000.
rows=5000; n=2000; cond_num=1e8; sparsity=0.7;
noise_level=0.5; lambda_frac=1e-2;
% Economical orthogonal basis: avoids allocating an enormous rows-by-rows U.
r=min(rows,n);
[U,~]=qr(randn(rows,r),0);
[V,~]=qr(randn(n,r),0);
s=logspace(0,-log10(cond_num),r)';
A=bsxfun(@times,U,s')*V';
A=bsxfun(@times,A,logspace(2,-2,n));
A(:,2:end)=A(:,2:end)+0.05*A(:,1);
xtrue=sprandn(n,1,sparsity);
b=A*xtrue+noise_level*randn(rows,1);
% IMPORTANT: normalize BEFORE selecting lambda and gamma.
coln=sqrt(sum(A.^2,1));coln(coln==0)=1;
A=bsxfun(@rdivide,A,coln);
L=normest(A)^2;
lambda=lambda_frac*L;
x0=zeros(n,1);gamma=1/(L+lambda);
delta=1e-4;maxit=2000;tol=1e-13;
csafe=1000; % Inf disables CROP consistency safeguard.

%% Display NRR problem information
[m, n] = size(A);
cond_A = cond(A);

fprintf('\n========== NRR PROBLEM INFORMATION ==========\n');
fprintf('Size of A: %d x %d\n', m, n);
fprintf('Target condition number: %.6e\n', cond_num);
fprintf('Actual condition number of A: %.6e\n', cond_A);
fprintf('Sparsity of x_true: %.2f%%\n', 100*nnz(xtrue)/n);
fprintf('Number of nonzeros in x_true: %d\n', nnz(xtrue));
fprintf('Noise level: %.6e\n', noise_level);
fprintf('Lambda fraction: %.6e\n', lambda_frac);
fprintf('Lambda used: %.12e\n', lambda);
fprintf('Lipschitz constant (L + lambda): %.12e\n', L + lambda);
fprintf('Stepsize gamma: %.12e\n', gamma);
fprintf('Maximum iterations: %d\n', maxit);
fprintf('Stopping tolerance: %.2e\n', tol);
fprintf('=============================================\n\n');

% Objective-reference strategy: best final objective among the ten methods.
method_names={'PG','FISTA','AA-PG (10)','AA-PG (20)', ...
 'CROP inexact (10)','CROP inexact (20)', ...
 'CROP exact (10)','CROP exact (20)','AdaPG','PGM-pc'};
info=cell(1,10);solutions=cell(1,10);accepted=nan(1,10);
[solutions{1},info{1}]=pg_nrr(A,b,lambda,x0,gamma,maxit,tol);
[solutions{2},info{2}]=fista_nrr(A,b,lambda,x0,gamma,maxit,tol);
[solutions{3},info{3},accepted(3)]=aa_pg_nrr(A,b,lambda,x0,gamma,10,maxit,tol);
[solutions{4},info{4},accepted(4)]=aa_pg_nrr(A,b,lambda,x0,gamma,20,maxit,tol);
[solutions{5},info{5},accepted(5)]=crop_pg_nrr(A,b,lambda,x0,gamma,delta,10,maxit,tol,csafe);
[solutions{6},info{6},accepted(6)]=crop_pg_nrr(A,b,lambda,x0,gamma,delta,20,maxit,tol,csafe);
[solutions{7},info{7},accepted(7)]=crop_pg_nrr_exact(A,b,lambda,x0,gamma,delta,10,maxit,tol,csafe);
[solutions{8},info{8},accepted(8)]=crop_pg_nrr_exact(A,b,lambda,x0,gamma,delta,20,maxit,tol,csafe);
[solutions{9},info{9}]=adapg_nrr(A,b,lambda,x0,gamma,maxit,tol,1.5);
[solutions{10},info{10}]=pgmpc_nrr(A,b,lambda,x0,gamma,maxit,tol,L+lambda);


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
    CPUTime_s(j) = rec.cpu_time;
    WallTime_s(j) = rec.elapsed_time;
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
    grad_x = A'*(A*xfinal-b)+lambda*xfinal;
    prox_x = max(0,xfinal-gamma_eval*grad_x);
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

outdir = fullfile(pwd,'results_NRR_PGResidual');
if ~exist(outdir,'dir'), mkdir(outdir); end
writetable(summary,fullfile(outdir,'NRR_10Methods_summary.csv'));

save(fullfile(outdir,'NRR_10Methods_results.mat'), ...
    'summary','info','solutions','phi_ref','gamma_eval', ...
    'lambda','lambda_frac','L', ...
    'cond_A','cond_num','m','n','sparsity', ...
    'noise_level','method_names_plot','-v7.3');


% Three standardized figures, logarithmic Y axes throughout.
% Objective gap: reference is best observed FINAL value, NOT a certified optimum.
plot_pg_comparison(info,method_names_plot,phi_ref,outdir,'NRR');

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
