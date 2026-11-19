%% Optimized ten-method l1 logistic comparison. Original plot appearance retained.
%clear;clc;
rng(0,'twister');
% Original common-reference experiment parameters:
rows=400;n=200;cond_num=1e5;sparsity=0.1;noise_level=0.5;lambda_frac=1e-2;
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
delta=1e-4;maxit=2000;tol=1e-14;csafe=10000; % c=Inf disables consistency safeguard.

%% CVX reference options (the ten original algorithms are unchanged)
run_cvx = true;              % false: skip the CVX solve
use_cvx_for_plots = true;    % use CVX objective when a valid CVX solution exists
cvx_precision_mode = 'best'; % 'default', 'high', or 'best'
cvx_relative_gap_tol = 1e-6; % normalized certified primal-dual gap
cvx_pg_residual_tol = 1e-5;  % normalized proximal-gradient residual
objective_check_tol = 1e-9; % consistency tolerance for objective values
cvx_compare_tol = 1e-8;     % relative tolerance for comparing two methods
% CVX may require substantial memory for the 6000-by-3000 dense matrix.


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
% Independent stable objective evaluation for every final iterate.
DirectObjective = nan(nmethods,1);
ObjectiveMismatch = nan(nmethods,1);
for j = 1:nmethods
    DirectObjective(j) = logistic_l1_objective(A,b,lambda,xx{j}(:));
    ObjectiveMismatch(j) = FinalObjective(j)-DirectObjective(j);
    mismatch_tol = objective_check_tol*max(1,abs(DirectObjective(j)));
    if ~isfinite(DirectObjective(j))
        warning('RLR:NonfiniteObjective', ...
            'Method %s has a nonfinite directly evaluated objective.',char(Method(j)));
    elseif abs(ObjectiveMismatch(j)) > mismatch_tol
        warning('RLR:ObjectiveMismatch', ...
            'Method %s: stored and direct objectives differ by %.3e.', ...
            char(Method(j)),ObjectiveMismatch(j));
    end
end
summary.DirectObjective = DirectObjective;
summary.ObjectiveMismatch = ObjectiveMismatch;
[phi_best, ref_idx] = min(DirectObjective);
fprintf('Best directly evaluated FINAL objective: %.16e (%s)\n', ...
    phi_best,char(Method(ref_idx)));

%% Independent CVX reference for logistic regression with l1 penalty.
% CVX is kept separate from all ten existing algorithm calls.
cvx_result = struct('status','Not run','objective',NaN,'cvx_optval',NaN, ...
    'time_s',NaN,'PGResidual',NaN,'primal_dual_gap',NaN, ...
    'relative_dual_gap',NaN,'dual_value',NaN,'dual_feasibility',NaN, ...
    'objective_discrepancy',NaN,'relative_to_best',NaN, ...
    'accepted_reference',false,'x',[]);
phi_ref = phi_best;
reference_source = 'best observed final objective (direct evaluation)';
if run_cvx
    if exist('cvx_begin','file') ~= 2
        warning('RLR:CVXMissing', ...
            'CVX is not on the MATLAB path; using best observed objective.');
        cvx_result.status = 'CVX not installed';
    else
        fprintf('\n========== CVX L1-RLR REFERENCE SOLVE ==========\n');
        cvx_result = solve_rlr_cvx_reference(A,b,lambda,gamma_eval,cvx_precision_mode);
        if strcmp(cvx_result.status,'Inaccurate/Solved')
            warning('RLR:CVXInaccurate', ...
                'CVX returned Inaccurate/Solved; independent diagnostics are required.');
        end
        if isfinite(cvx_result.objective)
            cvx_result.relative_to_best = ...
                (phi_best-cvx_result.objective)/max(1,abs(phi_best));
            compare_tolerance = cvx_compare_tol*max(1,abs(phi_best));
            if cvx_result.objective > phi_best + compare_tolerance
                warning('RLR:CVXAboveBest', ...
                    ['CVX objective is above the best solver by %.3e. ', ...
                     'CVX will NOT be used for plotting.'], ...
                     cvx_result.objective-phi_best);
            end
            % Certified numerical quality: check objective, dual gap,
            % PG stationarity, and agreement with the best algorithm.
            cvx_result.accepted_reference = ...
                strcmp(cvx_result.status,'Solved') && ...
                isfinite(cvx_result.relative_dual_gap) && ...
                cvx_result.relative_dual_gap >= -cvx_compare_tol && ...
                cvx_result.relative_dual_gap <= cvx_relative_gap_tol && ...
                isfinite(cvx_result.PGResidual) && ...
                cvx_result.PGResidual <= cvx_pg_residual_tol && ...
                abs(cvx_result.objective_discrepancy) <= ...
                    objective_check_tol*max(1,abs(cvx_result.objective)) && ...
                cvx_result.objective <= phi_best + compare_tolerance;
            if use_cvx_for_plots && cvx_result.accepted_reference
                % If roundoff produces an objective slightly above phi_best,
                % use the smaller directly observed primal objective.
                phi_ref = min(phi_best,cvx_result.objective);
                reference_source = 'CVX validated; minimum observed primal value';
            elseif use_cvx_for_plots
                warning('RLR:CVXReferenceRejected', ...
                    'CVX diagnostics did not pass; using best observed objective.');
            end
        end
    end
end
fprintf('CVX status: %s | objective: %.16e | elapsed: %.3fs\n', ...
    cvx_result.status,cvx_result.objective,cvx_result.time_s);
fprintf('CVX PG residual: %.6e | primal-dual gap: %.6e | relative gap: %.6e\n', ...
    cvx_result.PGResidual,cvx_result.primal_dual_gap, ...
    cvx_result.relative_dual_gap);
fprintf('CVX objective discrepancy: %.6e | dual constraint: %.6e\n', ...
    cvx_result.objective_discrepancy,cvx_result.dual_feasibility);
fprintf('CVX reference accepted: %d | Plot reference: %.16e (%s)\n', ...
    cvx_result.accepted_reference,phi_ref,reference_source);

% Preserve the original ten curves; CVX is an independent comparator.
summary.GapToCVX = nan(nmethods,1);
summary.RelativeGapToCVX = nan(nmethods,1);
if isfinite(cvx_result.objective)
    summary.GapToCVX = DirectObjective-cvx_result.objective;
    summary.RelativeGapToCVX = ...
        summary.GapToCVX/max(1,abs(cvx_result.objective));
    disp(summary(:,{'Method','DirectObjective','GapToCVX', ...
        'RelativeGapToCVX','PGResidual'}));
end

outdir = fullfile(pwd,'results_L1_RLR_PGResidual');
if ~exist(outdir,'dir'), mkdir(outdir); end
writetable(summary,fullfile(outdir,'L1_RLR_10Methods_summary.csv'));

save(fullfile(outdir,'L1_RLR_10Methods_results.mat'), ...
    'summary','infos','xx','phi_ref','phi_best','reference_source', ...
    'cvx_result','run_cvx','use_cvx_for_plots','gamma_eval', ...
    'cvx_relative_gap_tol','cvx_pg_residual_tol', ...
    'objective_check_tol','cvx_compare_tol', ...
    'lambda','lambda_frac','lambda_max','L', ...
    'cond_A','cond_num','m','n','sparsity', ...
    'noise_level','method_names_plot','-v7.3');


% Three standardized figures, logarithmic Y axes throughout.
% Objective gap: CVX numerical reference if available; otherwise best observed.
% CVX is not a certified optimum unless a sufficiently small gap is verified.
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


% ================================================================
% Independent helper: CVX is scoped inside its own function.
% ================================================================
function result = solve_rlr_cvx_reference(A,b,lambda,gamma_eval,precision_mode)
    [m,n] = size(A);
    result = struct('status','Not run','objective',NaN,'cvx_optval',NaN, ...
        'time_s',NaN,'PGResidual',NaN,'primal_dual_gap',NaN, ...
        'relative_dual_gap',NaN,'dual_value',NaN,'dual_feasibility',NaN, ...
        'objective_discrepancy',NaN,'relative_to_best',NaN, ...
        'accepted_reference',false,'x',[]);
    t_start = tic;
    try
        % Prevent an unfinished CVX model in this function's scope.
        cvx_clear
        cvx_begin quiet
            cvx_precision(precision_mode)
            variable x_cvx(n)
            minimize( sum(log_sum_exp([zeros(m,1), -b.*(A*x_cvx)],2))/m ...
                      + lambda*norm(x_cvx,1) )
        cvx_end
        result.time_s = toc(t_start);
        result.status = cvx_status;
        result.cvx_optval = cvx_optval;
        success = (strcmp(cvx_status,'Solved') || ...
                   strcmp(cvx_status,'Inaccurate/Solved')) && ...
                   isfinite(cvx_optval) && all(isfinite(x_cvx(:)));
        if ~success
            warning('RLR:CVXStatus','CVX status %s; no reference accepted.',cvx_status);
            return
        end
        result.x = full(x_cvx(:));
        result.objective = logistic_l1_objective(A,b,lambda,result.x);
        result.objective_discrepancy = result.objective-result.cvx_optval;
        z = b.*(A*result.x);
        q = exp(-max(z,0))./(1+exp(-abs(z)));
        grad = -(A'*(b.*q))/m;
        v = result.x-gamma_eval*grad;
        prox = sign(v).*max(abs(v)-gamma_eval*lambda,0);
        result.PGResidual = norm((result.x-prox)/gamma_eval) ...
                          /max(1,norm(result.x));

        % Fenchel dual: maximize -mean(q log q + (1-q) log(1-q)),
        % s.t. 0 <= q <= 1 and ||A'*(b.*q)/m||_inf <= lambda.
        if lambda > 0
            dual_norm = norm(A'*(b.*q),Inf)/m;
            scale = min(1,lambda/max(dual_norm,realmin));
            scale = scale*(1-1e-12); % slack for floating-point feasibility
            q_dual = max(0,min(1,scale*q));
            result.dual_feasibility = norm(A'*(b.*q_dual),Inf)/m;
            interior = q_dual>0 & q_dual<1;
            entropy = zeros(m,1);
            entropy(interior) = q_dual(interior).*log(q_dual(interior)) ...
                + (1-q_dual(interior)).*log1p(-q_dual(interior));
            result.dual_value = -sum(entropy)/m;
            result.primal_dual_gap = result.objective-result.dual_value;
            result.relative_dual_gap = result.primal_dual_gap ...
                                      /max(1,abs(result.objective));
            if result.dual_feasibility > lambda*(1+1e-10)
                warning('RLR:DualFeasibility', ...
                    'Computed dual point violates the l1 constraint.');
                result.relative_dual_gap = NaN;
            end
        end
    catch ME
        result.time_s = toc(t_start);
        result.status = ['Error: ' ME.message];
        warning('RLR:CVXFailure','CVX failed: %s',ME.message);
        % Do not mask failures by claiming a successful reference.
        try
            cvx_clear
        catch
        end
    end
end

function value = logistic_l1_objective(A,b,lambda,x)
    z = b.*(A*x);
    % softplus(-z) = max(0,-z)+log(1+exp(-abs(z)))
    value = mean(max(0,-z)+log1p(exp(-abs(z))))+lambda*norm(x,1);
end
