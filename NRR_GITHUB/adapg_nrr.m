function [x,info] = adapg_nrr(A,b,lambda,x0,gamma,maxit,tol,q)
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
if nargin < 8 || isempty(q), q=1.5; end
if ~isscalar(q) || q<1 || q>2, error('q must lie in [1,2]'); end
cpu0=cputime; wall0=tic;
[grad,obj,backend]=nrr_operators(A,b,lambda);
setup_time=toc(wall0); solver_start=tic;
prev=x0; gprev=grad(prev); phi0=obj(prev);
x=max(prev-gamma*gprev,0); gx=grad(x); phi=obj(x);
f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1); res_hist=zeros(1,maxit);
gamma_hist=zeros(1,maxit); ell_hist=zeros(1,maxit); L_hist=zeros(1,maxit);
f_hist(1)=phi0; f_hist(2)=phi; t_hist(2)=toc(solver_start);
res_hist(1)=abs(phi-phi0)/(1+abs(phi0)); gamma_hist(1)=gamma;
gprevstep=gamma; gcur=gamma; k=1;
if res_hist(1)>=tol
  for k=2:maxit
    dx=x-prev; dg=gx-gprev; dx2=dx'*dx;
    if dx2>0, ell=real(dx'*dg)/dx2; Lloc=sqrt(real(dg'*dg)/dx2);
    else, ell=0; Lloc=0; end
    growth=sqrt(1/q+gcur/gprevstep);
    rad=2*max(gcur^2*Lloc^2-(2-q)*gcur*ell+(1-q),0);
    if rad==0, curvature=Inf; else, curvature=1/sqrt(rad); end
    gnext=gcur*min(growth,curvature);
    if ~isfinite(gnext) || gnext<=0, error('Invalid AdaPG step at %d',k); end
    xnew=max(x-gnext*gx,0); phinew=obj(xnew);
    res_hist(k)=abs(phinew-phi)/(1+abs(phi));
    f_hist(k+1)=phinew; t_hist(k+1)=toc(solver_start);
    gamma_hist(k)=gnext; ell_hist(k)=ell; L_hist(k)=Lloc;
    prev=x; gprev=gx; x=xnew; phi=phinew;
    gprevstep=gcur; gcur=gnext;
    if res_hist(k)<tol, break; end
    if k<maxit, gx=grad(x); end
  end
end
info.k=k; info.f_hist=f_hist(1:k+1); info.t_hist=t_hist(1:k+1);
info.res_hist=res_hist(1:k); info.gamma_hist=gamma_hist(1:k);
info.ell_hist=ell_hist(1:k); info.L_hist=L_hist(1:k);
info.elapsed_time=toc(wall0); info.cpu_time=cputime-cpu0;
info.setup_time=setup_time; info.backend=backend; info.final_objective=phi; info.q=q;
fprintf('AdaPG             iterations=%5d  time=%.4fs  objective=%.9e\n',k,info.elapsed_time,phi);
end
