function [x,info] = pgmpc_nrr(A,b,lambda,x0,gamma,maxit,tol,L,opts)
% PGM-pc Algorithm 5.1, Chang et al., JCAM 485 (2026), 117536.
% min_{x>=0} 0.5*norm(A*x-b)^2 + (lambda/2)*norm(x)^2.
% gamma = initial stepsize, L >= norm(A)^2 + lambda (smoothness bound).
% Relative objective-change stopping, matching pg_nrr_optimized.
% opts.alpha=.99, opts.theta0=.99, opts.rho=.7, opts.lambda_hat=1e8,
% opts.max_linesearch=200, opts.check_tolerance=1e-12.
if nargin<9 || isempty(opts), opts=struct(); end
if nargin<8 || isempty(L), L=normest(A)^2+lambda; end
if nargin<7 || isempty(tol), tol=1e-8; end
if nargin<6 || isempty(maxit), maxit=1000; end
if nargin<4 || isempty(x0), x0=zeros(size(A,2),1); end
if nargin<5 || isempty(gamma), gamma=1/L; end
alpha=opt(opts,'alpha',0.99); theta=opt(opts,'theta0',0.99);
shrink=opt(opts,'rho',0.7); lambda_hat=opt(opts,'lambda_hat',1e8);
max_ls=opt(opts,'max_linesearch',200);
check_tol=opt(opts,'check_tolerance',1e-12);
[m,n]=size(A); b=b(:); x=x0(:);
assert(numel(b)==m && numel(x)==n,'Dimension mismatch');
assert(all(x>=0),'Initial point x0 must be feasible (nonnegative)');
assert(isscalar(lambda) && isfinite(lambda) && lambda>=0,'Invalid lambda');
assert(isscalar(L) && isfinite(L) && L>0 && isfinite(gamma) && gamma>0,'Invalid L/gamma');
assert(alpha>0 && alpha<1 && theta>0 && theta<1 && shrink>0 && shrink<1,'Invalid options');
assert(isfinite(lambda_hat) && lambda_hat>0 && maxit>=1 && maxit==floor(maxit),'Invalid cap/maxit');
assert(isfinite(tol) && tol>=0,'Invalid tol');
assert(max_ls>=1 && max_ls==floor(max_ls) && check_tol>=0,'Invalid line search settings');
% Avoid forming A'*A for n=5000. Cache A*x for evaluating both f and grad.
clock=tic; cpu0=cputime;
f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1);
res_hist=zeros(1,maxit); lambda_hist=zeros(1,maxit);
correction_trials=0; backtrack_steps=0; gradient_evals=0;
[phi,grad]=eval_fg(x); f_hist(1)=phi;
% Algorithm 5.1 Step 0: obtain x1 using lambda0.
lam_prev=gamma;
xnew=max(x-lam_prev*grad,0);
[phinew,gnew]=eval_fg(xnew);
res=abs(phinew-phi)/(1+abs(phi));
xold=x; gold=grad;
x=xnew; grad=gnew; phi=phinew;
f_hist(2)=phi; res_hist(1)=res; lambda_hist(1)=lam_prev;
t_hist(2)=toc(clock); kfinal=1;
rho_prev=sqrt(2); w_prev=1; delta_prev=1;
% Avoid claiming true convergence solely due to zero objective difference:
% termination matches existing comparison's objective-change criterion.
for k=1:maxit-1
    if res<tol || isequal(x,xold), break; end
    % Algorithm 5.1, Step 1, Eq. (41).
    dp=x-xold; gp=grad-gold;
    dp2=dot(dp,dp); gp2=dot(gp,gp);
    if dp2==0, break; end
    Lk=sqrt(gp2/dp2);
    denom=1-alpha/(2*L*lam_prev);
    if denom<=0
        error('PGM-pc:Eq41','Equation (41) requires lambda_(k-1)>alpha/(2L).');
    end
    if Lk==0, cap=Inf; else, cap=alpha/(2*Lk*sqrt(denom)); end
    lam=min([rho_prev*lam_prev,cap,lambda_hat]);
    if ~isfinite(lam) || lam<=0, error('PGM-pc:Step','Invalid predicted step'); end
    xnew=max(x-lam*grad,0);
    [phinew,gnew]=eval_fg(xnew);
    d=xnew-x; d2=dot(d,d);
    % Eq. (12) and improved correction Eq. (40).
    regdiff=0; % g is the indicator of R_+^n: both iterates are feasible.
    a=(dot(grad,-d)-regdiff)/lam_prev;
    bc=dot(gp,-d)-d2/lam_prev;
    c=alpha*(dp2+d2)/4;
    lhs=(a-gp2/(2*L*lam_prev))*lam^2+bc*lam;
    w=1;
    if lhs>c+check_tol*max([1,abs(lhs),abs(c)])
        w=0; correction_trials=correction_trials+1;
        % Algorithm 3.1, gradient-only linesearch Eq. (6).
        j=0;
        while dot(gnew-grad,d)>d2/lam + check_tol*max(1,abs(d2/lam))
            j=j+1;
            if j>max_ls, error('PGM-pc:Linesearch','Line search exceeded limit'); end
            lam=shrink*lam;
            xnew=max(x-lam*grad,0);
            [phinew,gnew]=eval_fg(xnew);
            d=xnew-x; d2=dot(d,d);
            backtrack_steps=backtrack_steps+1;
        end
    end
    % Algorithm 5.1, Step 3.
    delta=lam/lam_prev;
    theta=max(theta,1/sqrt(1+delta));
    if w_prev==1 && w==0
        rho_prev=sqrt(theta);
    elseif w_prev==0
        rho_prev=sqrt(1+theta*delta_prev/delta);
    else
        rho_prev=sqrt(1+theta*delta);
    end
    delta_prev=delta; w_prev=w; lam_prev=lam;
    res=abs(phinew-phi)/(1+abs(phi));
    xold=x; gold=grad; x=xnew; grad=gnew; phi=phinew;
    kfinal=k+1;
    f_hist(kfinal+1)=phi; res_hist(kfinal)=res;
    lambda_hist(kfinal)=lam; t_hist(kfinal+1)=toc(clock);
end
info.k=kfinal; info.f_hist=f_hist(1:kfinal+1);
info.t_hist=t_hist(1:kfinal+1); info.res_hist=res_hist(1:kfinal);
info.lambda_hist=lambda_hist(1:kfinal);
info.final_objective=phi; info.final_residual=res_hist(kfinal);
info.elapsed_time=toc(clock); info.cpu_time=cputime-cpu0;
info.gradient_evals=gradient_evals; info.objective_evals=gradient_evals;
info.correction_trials=correction_trials; info.backtrack_steps=backtrack_steps;
info.setup_time=0; info.gradient_mode='matrix-free';
info.converged=info.final_residual<tol;
fprintf('PGM-pc: iterations=%d, wall=%.4fs, CPU=%.4fs, objective=%.9e, backtracks=%d\n', ...
  kfinal,info.elapsed_time,info.cpu_time,phi,backtrack_steps);
    function [F,G]=eval_fg(u)
        e=A*u-b;
        F=0.5*dot(e,e)+0.5*lambda*dot(u,u);
        G=A'*e+lambda*u;
        gradient_evals=gradient_evals+1;
    end
end
function v=opt(s,key,d)
if isfield(s,key) && ~isempty(s.(key)), v=s.(key); else, v=d; end
end
