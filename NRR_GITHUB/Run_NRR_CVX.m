%% Ten-method NRR comparison (optimized)
% Keep this script and nrr_operators.m in the same folder as the six solvers.
%clear; clc; 
rng(0,'twister');
% Adjust problem size as needed; original experiment: rows=5000, cols=2000.
rows=500; n=300; cond_num=1e8; sparsity=0.7;
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
delta=1e-4;maxit=2000;tol=1e-15;
csafe=1000; % Inf disables CROP consistency safeguard.

%% CVX reference options (all ten original solvers remain unchanged)
run_cvx = true;              % false: skip CVX
use_cvx_for_plots = true;    % use CVX objective as phi_ref if valid
cvx_precision_mode = 'best'; % 'default', 'high', or 'best'
cvx_relative_gap_tol = 1e-6; % certificate threshold for CVX plot eligibility
cvx_pg_residual_tol = 1e-5; % stationarity threshold for CVX plot eligibility
objective_check_tol = 1e-9; % check objective recorded by each algorithm
% This large dense quadratic program may need substantial RAM in CVX.

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
DirectObjective = nan(nmethods,1);
ObjectiveMismatch = nan(nmethods,1);
NonnegViolation = nan(nmethods,1);

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
    % Re-evaluate all methods with EXACTLY the same NRR objective as CVX.
    residual_j = A*xfinal-b;
    DirectObjective(j) = 0.5*(residual_j'*residual_j) ...
                       + 0.5*lambda*(xfinal'*xfinal);
    ObjectiveMismatch(j) = FinalObjective(j)-DirectObjective(j);
    NonnegViolation(j) = max(0,-min(xfinal));
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
    FinalRelChange,PGResidual,AcceptedPct,StopMet, ...
    DirectObjective,ObjectiveMismatch,NonnegViolation);
disp(summary);
fprintf('\nFinal normalized PG residuals (common gamma = %.12e):\n',gamma_eval);
disp(summary(:,{'Method','Iterations','CPUTime_s','FinalObjective', ...
    'PGResidual','StopMet'}));
[phi_best, ref_idx] = min(FinalObjective);
fprintf('Common best FINAL objective: %.16e (%s)\n',phi_best,Method(ref_idx));
[phi_best_direct, best_direct_idx] = min(DirectObjective);
recorded_objectives_consistent = all(isfinite(DirectObjective)) && ...
    all(isfinite(FinalObjective)) && all(NonnegViolation <= 1e-8) && ...
    all(abs(ObjectiveMismatch) <= objective_check_tol .* ...
        max(1,abs(DirectObjective)));
fprintf('Best DIRECT objective: %.16e (%s)\n', ...
    phi_best_direct,Method(best_direct_idx));
fprintf('Max stored/direct difference: %.6e\n',max(abs(ObjectiveMismatch)));
fprintf('Max nonnegative violation: %.6e\n',max(NonnegViolation));
if ~recorded_objectives_consistent
    warning('NRR:ObjectiveMismatch', ...
        ['Stored objectives and directly evaluated NRR objectives disagree, ' ...
         'or a solver returned an infeasible point. Check solver definitions.']);
end

%% Independent CVX reference for NRR
% min_{x >= 0} 0.5*||A*x-b||_2^2 + 0.5*lambda*||x||_2^2.
% Important: the original NRR objective has NO division by m.
% The CVX block is INSIDE a separate local function to prevent stale CVX
% models in the script/base workspace from causing scope warnings.
cvx_result = struct('status','Not run','objective',NaN,'time_s',NaN, ...
    'PGResidual',NaN,'primal_dual_gap',NaN,'relative_duality_gap',NaN, ...
    'dual_value',NaN,'relative_to_best',NaN,'min_x',NaN, ...
    'cvx_optval',NaN,'cvx_objective_difference',NaN, ...
    'certified',false,'x',[]);
phi_ref = phi_best;
reference_source = 'best observed final objective';
if run_cvx
    if exist('cvx_begin','file') ~= 2
        warning('NRR:CVXNotFound', ...
            'CVX is not on the MATLAB path; using best observed objective.');
        cvx_result.status = 'CVX not installed';
    else
        fprintf('\n========== CVX NRR REFERENCE SOLVE ==========\n');
        t_cvx = tic;
        try
            [x_cvx, cvx_status_local, cvx_optval_local] = ...
                solve_nrr_reference_cvx(A,b,lambda,cvx_precision_mode);
            cvx_result.time_s = toc(t_cvx);
            cvx_result.status = char(cvx_status_local);
            cvx_result.cvx_optval = cvx_optval_local;
            is_valid_cvx = (strcmp(cvx_status_local,'Solved') || ...
                strcmp(cvx_status_local,'Inaccurate/Solved')) && ...
                isfinite(cvx_optval_local) && ...
                isnumeric(x_cvx) && numel(x_cvx)==n && ...
                all(isfinite(x_cvx(:)));
            if is_valid_cvx
                cvx_result.x = full(x_cvx(:));
                cvx_result.min_x = min(cvx_result.x);
                % CVX is approximate: evaluate at its actual returned point.
                % Never silently clip negative coordinates in this check.
                r_cvx = A*cvx_result.x-b;
                primal_cvx = 0.5*(r_cvx'*r_cvx) + ...
                             0.5*lambda*(cvx_result.x'*cvx_result.x);
                grad_cvx = A'*r_cvx+lambda*cvx_result.x;
                prox_cvx = max(0,cvx_result.x-gamma_eval*grad_cvx);
                cvx_result.PGResidual = ...
                    norm((cvx_result.x-prox_cvx)/gamma_eval) ...
                    /max(1,norm(cvx_result.x));
                % Valid dual lower bound when lambda>0:
                % D(y) = -||y||^2/2 - b'*y - ||min(A'*y,0)||^2/(2*lambda).
                if lambda > 0
                    y_dual = r_cvx;
                    negative_aty = min(A'*y_dual,0);
                    cvx_result.dual_value = ...
                        -0.5*(y_dual'*y_dual)-b'*y_dual ...
                        -(negative_aty'*negative_aty)/(2*lambda);
                    cvx_result.primal_dual_gap = ...
                        primal_cvx-cvx_result.dual_value;
                    cvx_result.relative_duality_gap = ...
                        cvx_result.primal_dual_gap/max(1,abs(primal_cvx));
                end
                cvx_result.objective = primal_cvx;
                cvx_result.cvx_objective_difference = ...
                    cvx_optval_local-primal_cvx;
                cvx_result.relative_to_best = ...
                    (phi_best_direct-primal_cvx)/max(1,abs(primal_cvx));
                if phi_best_direct < cvx_result.dual_value - ...
                        1e-9*max(1,abs(cvx_result.dual_value))
                    warning('NRR:DualBoundInconsistency', ...
                        ['Best solver objective is below the independently computed ' ...
                         'dual lower bound. Check normalization, feasibility or arithmetic.']);
                end
                feasibility_ok = cvx_result.min_x >= -1e-8;
                gap_ok = cvx_result.primal_dual_gap >= ...
                    -1e-10*max(1,abs(primal_cvx)) && ...
                    cvx_result.relative_duality_gap <= cvx_relative_gap_tol;
                stationarity_ok = cvx_result.PGResidual <= cvx_pg_residual_tol;
                cvx_result.certified = feasibility_ok && gap_ok && ...
                    stationarity_ok && strcmp(cvx_status_local,'Solved');
                if strcmp(cvx_status_local,'Inaccurate/Solved')
                    warning('NRR:CVXInaccurate', ...
                        'CVX returned Inaccurate/Solved; do not treat it as a certified reference.');
                end
                % A solver point lower than CVX by more than rounding tolerance
                % calls for investigation, never blindly choose CVX for plots.
                discrepancy = primal_cvx-phi_best_direct;
                discrepancy_tol = 1e-9*max(1,abs(phi_best_direct));
                if discrepancy > discrepancy_tol
                    warning('NRR:CVXObjectiveHigher', ...
                        ['CVX objective is larger than the best directly evaluated ' ...
                         'solver objective by %.6e (relative %.6e).'], ...
                         discrepancy, discrepancy/max(1,abs(phi_best_direct)));
                end
                if ~feasibility_ok
                    warning('NRR:CVXFeasibility', ...
                        'CVX violates x >= 0: min(x) = %.6e.',cvx_result.min_x);
                end
                if ~gap_ok || ~stationarity_ok
                    warning('NRR:CVXAccuracy', ...
                        ['CVX accuracy check: relative duality gap %.3e; ' ...
                         'PG residual %.3e. Reference not certified to thresholds.'], ...
                         cvx_result.relative_duality_gap,cvx_result.PGResidual);
                end
                if use_cvx_for_plots
                    if cvx_result.certified && recorded_objectives_consistent && ...
                            primal_cvx <= phi_best + discrepancy_tol
                        % Use the lower objective to prevent negative plot gaps.
                        phi_ref = min(phi_best,primal_cvx);
                        if primal_cvx < phi_best
                            reference_source = 'CVX (checked primal-dual reference)';
                        else
                            reference_source = 'best observed (CVX agrees numerically)';
                        end
                    else
                        warning('NRR:CVXPlotFallback', ...
                            'CVX is not eligible as plot reference; using best observed objective.');
                    end
                end
            else
                warning('NRR:CVXNotSolved', ...
                    'CVX status: %s; keeping best observed reference.',cvx_result.status);
            end
        catch ME
            cvx_result.time_s = toc(t_cvx);
            cvx_result.status = ['Error: ' ME.message];
            warning('NRR:CVXFailed','CVX reference failed: %s',ME.message);
            fprintf('%s\n',getReport(ME,'extended','hyperlinks','off'));
        end
    end
end
fprintf('CVX status: %s | objective: %.16e | elapsed: %.3fs\n', ...
    cvx_result.status,cvx_result.objective,cvx_result.time_s);
fprintf('CVX PG residual: %.6e | primal-dual gap: %.6e\n', ...
    cvx_result.PGResidual,cvx_result.primal_dual_gap);
fprintf('CVX relative duality gap: %.6e | min(x): %.6e | accepted: %d\n', ...
    cvx_result.relative_duality_gap,cvx_result.min_x,cvx_result.certified);
fprintf('Plot reference: %.16e (%s)\n',phi_ref,reference_source);

% Compare the original ten methods against CVX; do not plot CVX as an 11th method.
summary.GapToCVX = nan(nmethods,1);
summary.RelativeGapToCVX = nan(nmethods,1);
if isfinite(cvx_result.objective) && ~isempty(cvx_result.x)
    summary.GapToCVX = FinalObjective-cvx_result.objective;
    summary.RelativeGapToCVX = ...
        summary.GapToCVX/max(1,abs(cvx_result.objective));
    disp(summary(:,{'Method','FinalObjective','GapToCVX', ...
        'RelativeGapToCVX','PGResidual'}));
end

outdir = fullfile(pwd,'results_NRR_PGResidual');
if ~exist(outdir,'dir'), mkdir(outdir); end
writetable(summary,fullfile(outdir,'NRR_10Methods_summary.csv'));

save(fullfile(outdir,'NRR_10Methods_results.mat'), ...
    'summary','info','solutions','phi_ref','phi_best','reference_source', ...
    'cvx_result','run_cvx','use_cvx_for_plots','gamma_eval', ...
    'cvx_relative_gap_tol','cvx_pg_residual_tol','objective_check_tol', ...
    'recorded_objectives_consistent','phi_best_direct', ...
    'lambda','lambda_frac','L', ...
    'cond_A','cond_num','m','n','sparsity', ...
    'noise_level','method_names_plot','-v7.3');

% Original three plots, with a CVX reference if enabled and valid.
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
            xlabel('Wall-clock time (s)','Interpreter','latex','FontSize',18);
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

% CVX in an isolated function scope prevents stale base-workspace models.
function [xsol,status,optval] = solve_nrr_reference_cvx(A,b,lambda,precision_mode)
    if ~any(strcmp(char(precision_mode),{'default','high','best'}))
        error('NRR:InvalidCVXPrecision', ...
            'cvx_precision_mode must be default, high, or best.');
    end
    nvars = size(A,2);
    cvx_clear
    cvx_begin quiet
        cvx_precision(precision_mode)
        variable z(nvars) nonnegative
        minimize(0.5*sum_square(A*z-b) + 0.5*lambda*sum_square(z))
    cvx_end
    xsol = full(z(:));
    status = cvx_status;
    optval = cvx_optval;
end
