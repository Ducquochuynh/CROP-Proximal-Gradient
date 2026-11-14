function [x,info]=adapg_logistic_l1(A,b,lambda,x0,gamma,maxit,tol,q)
% Universal AdaPG q,q/2. Initialization counts as one proximal step.
if nargin<8 || isempty(q), q=1.5; end
if ~isscalar(q) || q<1 || q>2 || ~isfinite(q), error('q must be in [1,2]'); end
% Consistent information fields: info.k proximal steps, info.f_hist,
% info.t_hist (elapsed wall seconds), info.res_hist, info.elapsed_time,
% info.cpu_time, info.final_objective. History starts with x0 at index 1.
if nargin < 7 || isempty(tol), tol = 1e-8; end
if nargin < 6 || isempty(maxit), maxit = 1000; end
if nargin < 4 || isempty(x0), x0=zeros(size(A,2),1); end
[mA,n]=size(A); b=b(:);x0=x0(:);
if numel(b)~=mA || numel(x0)~=n || any(abs(b)~=1), error('Dimension or label mismatch'); end
if ~isscalar(lambda) || ~isfinite(lambda) || lambda<0, error('Invalid lambda'); end
if ~isscalar(maxit) || maxit<1 || maxit~=floor(maxit), error('maxit must be integer >=1'); end
if ~isscalar(tol) || ~isfinite(tol) || tol<=0, error('Invalid tol'); end
if nargin < 5 || isempty(gamma), gamma=4*mA/max(normest(A)^2,eps); end
if ~isscalar(gamma) || ~isfinite(gamma) || gamma<=0, error('Invalid gamma'); end
obj=@(u) logistic_l1_ops('obj',A,b,u,lambda);
grad=@(u) logistic_l1_ops('grad',A,b,u,lambda);
smooth=@(u) logistic_l1_ops('smooth',A,b,u,lambda);
prox=@(u) sign(u).*max(abs(u)-gamma*lambda,0);
f_hist=zeros(1,maxit+1);t_hist=zeros(1,maxit+1);res_hist=zeros(1,maxit);
x=x0;phik=obj(x);f_hist(1)=phik;
clock=tic;cpustart=cputime;k_final=0;
gamma_hist=nan(1,maxit+1);ell_hist=nan(1,maxit+1);L_hist=nan(1,maxit+1);
x_prev=x;grad_prev=grad(x_prev);
x=sign(x_prev-gamma*grad_prev).*max(abs(x_prev-gamma*grad_prev)-gamma*lambda,0);
phinew=obj(x);k_final=1;f_hist(2)=phinew;t_hist(2)=toc(clock);
res_hist(1)=abs(phinew-phik)/(1+abs(phik));gamma_hist(1)=gamma;
grad_cur=grad(x);gprev=gamma;gcur=gamma;phik=phinew;
for k=2:maxit
    dx=x-x_prev;dg=grad_cur-grad_prev;dx2=dx'*dx;
    if dx2>0, ell=(dx'*dg)/dx2;Lloc=norm(dg)/sqrt(dx2);
    else, ell=0;Lloc=0; end
    growth=sqrt(1/q+gcur/gprev);
    v=2*max(gcur^2*Lloc^2-(2-q)*gcur*ell+1-q,0);
    if v==0, curvature=Inf; else, curvature=1/sqrt(v); end
    gnext=gcur*min(growth,curvature);
    if ~isfinite(gnext) || gnext<=0, warning('AdaPG:BadStep','Terminating due to invalid stepsize');break;end
    xnew=sign(x-gnext*grad_cur).*max(abs(x-gnext*grad_cur)-gnext*lambda,0);
    phi=obj(xnew);res=abs(phi-phik)/(1+abs(phik));
    x_prev=x;grad_prev=grad_cur;x=xnew;phik=phi;k_final=k;
    f_hist(k+1)=phi;res_hist(k)=res;t_hist(k+1)=toc(clock);
    gamma_hist(k)=gnext;ell_hist(k)=ell;L_hist(k)=Lloc;
    if res<tol,break;end
    grad_cur=grad(x);gprev=gcur;gcur=gnext;
end
info.k=k_final;
info.f_hist=f_hist(1:k_final+1);
info.t_hist=t_hist(1:k_final+1);
info.res_hist=res_hist(1:k_final);
info.final_objective=info.f_hist(end);
info.elapsed_time=toc(clock);
info.cpu_time=cputime-cpustart;
info.setup_time=0;
info.converged= k_final>0 && info.res_hist(end)<tol;
info.q=q;info.gamma_hist=gamma_hist(1:k_final);
info.ell_hist=ell_hist(1:k_final);info.L_hist=L_hist(1:k_final);
end
