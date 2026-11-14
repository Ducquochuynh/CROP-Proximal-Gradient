function [x,info] = pg_nrr(A,b,lambda,x0,gamma,maxit,tol)
% Optimized NRR solver: min_{x>=0} 0.5*||A*x-b||^2 + lambda/2*||x||^2.
% info.k = total projected-gradient/proximal updates, including initialization.
% info.t_hist = elapsed solver time at each recorded objective.
% info.elapsed_time = total wall time including setup; info.cpu_time = process CPU seconds.
if nargin < 7 || isempty(tol), tol=1e-8; end
if nargin < 6 || isempty(maxit), maxit=1000; end
if nargin < 5, gamma=[]; end
[rows,n]=size(A); b=b(:);
if numel(b)~=rows, error('b dimension mismatch'); end
if nargin < 4 || isempty(x0), x0=zeros(n,1); end
x0=x0(:);
if numel(x0)~=n || lambda<0 || maxit<1 || maxit~=floor(maxit), error('Invalid inputs'); end
if isempty(gamma), gamma=1/(normest(A)^2+lambda); end
if ~isfinite(gamma) || gamma<=0, error('gamma must be positive'); end
cpu0=cputime; wall0=tic;
[grad,obj,backend]=nrr_operators(A,b,lambda);
setup_time=toc(wall0); solver_start=tic;
x=x0; phi=obj(x);
f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1); res_hist=zeros(1,maxit);
f_hist(1)=phi;
for k=1:maxit
    xnew=max(x-gamma*grad(x),0);
    phinew=obj(xnew);
    res_hist(k)=abs(phinew-phi)/(1+abs(phi));
    f_hist(k+1)=phinew; t_hist(k+1)=toc(solver_start);
    x=xnew; phi=phinew;
    if res_hist(k)<tol, break; end
end
info.k=k; info.f_hist=f_hist(1:k+1); info.t_hist=t_hist(1:k+1);
info.res_hist=res_hist(1:k); info.elapsed_time=toc(wall0);
info.cpu_time=cputime-cpu0; info.setup_time=setup_time;
info.backend=backend; info.final_objective=phi;
fprintf('%-17s iterations=%5d  time=%.4fs  objective=%.9e\n', ...
    'PG',k,info.elapsed_time,phi);
end
