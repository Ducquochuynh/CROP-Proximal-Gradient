%% Optimized ten-method l1 logistic comparison. Original plot appearance retained.
%clear;clc;
rng(0,'twister');
% Original common-reference experiment parameters:
rows=6000;n=3000;cond_num=1e5;sparsity=0.1;noise_level=0.5;lambda_frac=1e-2;
% Avoid QR(randn(n,n)): construct only rank=min(rows,n) orthonormal columns.
r=min(rows,n);[U,~]=qr(randn(rows,r),0);[V,~]=qr(randn(n,r),0);
s=logspace(0,-log10(cond_num),r)';
A=bsxfun(@times,U,s')*V';
A=bsxfun(@times,A,logspace(2,-2,n));
A(:,2:end)=A(:,2:end)+0.01*A(:,1);
xtrue=sprandn(n,1,sparsity);
b=sign(A*xtrue+noise_level*randn(rows,1));b(b==0)=1;
lambda_max=norm((A'*b)/(2*rows),Inf);lambda=lambda_frac*lambda_max;
x0=zeros(n,1);m = rows; L = norm(A, 2)^2 / (4*m);gamma=1/max(L,eps);
delta=1e-4;maxit=2000;tol=1e-13;csafe=10000; % c=Inf disables consistency safeguard.

%% Display problem information
[m, n] = size(A);
cond_A = cond(A);

fprintf('\n========== L1-RLR PROBLEM INFORMATION ==========\n');
fprintf('Size of A: %d x %d\n', m, n);
fprintf('Target condition number: %.6e\n', cond_num);
fprintf('Actual condition number of A: %.6e\n', cond_A);
fprintf('Sparsity of x_true: %.2f%%\n', 100*nnz(xtrue)/n);
fprintf('Number of nonzeros in x_true: %d\n', nnz(xtrue));
fprintf('Noise level: %.6e\n', noise_level);
fprintf('Lambda fraction: %.6e\n', lambda_frac);
fprintf('Lambda max: %.12e\n', lambda_max);
fprintf('Lambda used: %.12e\n', lambda);
fprintf('Lipschitz constant L: %.12e\n', L);
fprintf('Stepsize gamma: %.12e\n', gamma);
fprintf('Maximum iterations: %d\n', maxit);
fprintf('Stopping tolerance: %.2e\n', tol);
fprintf('================================================\n\n');
method_names={'PG','FISTA','AA-PG (m=10)','AA-PG (m=20)', ...
 'CROP-PG-Inexact (m=10)','CROP-PG-Inexact (m=20)', ...
 'CROP-PG-Exact (m=10)','CROP-PG-Exact (m=20)','AdaPG','PGM-pc'};
infos=cell(1,10);xx=cell(1,10);acc=nan(10,1);
[xx{1},infos{1}]=pg_logistic_l1(A,b,lambda,x0,gamma,maxit,tol);
[xx{2},infos{2}]=fista_logistic_l1(A,b,lambda,x0,gamma,maxit,tol);
[xx{3},infos{3},acc(3)]=aa_pg_logistic_l1(A,b,lambda,x0,gamma,10,maxit,tol);
[xx{4},infos{4},acc(4)]=aa_pg_logistic_l1(A,b,lambda,x0,gamma,20,maxit,tol);
[xx{5},infos{5},acc(5)]=crop_pg_logistic_l1(A,b,lambda,x0,gamma,delta,10,maxit,tol,csafe);
[xx{6},infos{6},acc(6)]=crop_pg_logistic_l1(A,b,lambda,x0,gamma,delta,20,maxit,tol,csafe);
[xx{7},infos{7},acc(7)]=crop_pg_logistic_l1_exact(A,b,lambda,x0,gamma,delta,10,maxit,tol,csafe);
[xx{8},infos{8},acc(8)]=crop_pg_logistic_l1_exact(A,b,lambda,x0,gamma,delta,20,maxit,tol,csafe);
[xx{9},infos{9}]=adapg_logistic_l1(A,b,lambda,x0,gamma,maxit,tol,1.5);
[xx{10},infos{10}]=pgmpc_logistic_l1(A,b,lambda,x0,gamma,maxit,tol,L);


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
    rec = infos{j};
    xfinal = xx{j}(:);
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
        AcceptedPct(j) = acc(j);
    end
    z=b.*(A*xfinal);
    % Stable evaluation of 1/(1+exp(z)).
    sig_neg=exp(-max(z,0))./(1+exp(-abs(z)));
    grad_x=-(A'*(b.*sig_neg))/rows;
    v=xfinal-gamma_eval*grad_x;
    prox_x=sign(v).*max(abs(v)-gamma_eval*lambda,0);
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

outdir = fullfile(pwd,'results_L1_RLR_PGResidual');
if ~exist(outdir,'dir'), mkdir(outdir); end
writetable(summary,fullfile(outdir,'L1_RLR_10Methods_summary.csv'));

save(fullfile(outdir,'L1_RLR_10Methods_results.mat'), ...
    'summary','infos','xx','phi_ref','gamma_eval', ...
    'lambda','lambda_frac','lambda_max','L', ...
    'cond_A','cond_num','m','n','sparsity', ...
    'noise_level','method_names_plot','-v7.3');


% Three standardized figures, logarithmic Y axes throughout.
% Objective gap: reference is best observed FINAL value, NOT a certified optimum.
plot_pg_comparison(infos,method_names_plot,phi_ref,outdir,'L1_RLR');

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
