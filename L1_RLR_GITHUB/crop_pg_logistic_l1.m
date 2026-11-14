function [x,info,acc]=crop_pg_logistic_l1(A,b,lambda,x0,gamma,delta,m,maxit,tol,c)
% CROP-PG logistic with optional asymptotic consistency safeguard.
% c=Inf disables safeguard; finite c enforces ||yCr-z_k||<=c||xk-xprev||.
% Inexact Hk*alpha residual on accepted steps.
if nargin<10 || isempty(c),c=1;end
if nargin<9 || isempty(tol),tol=1e-8;end
if nargin<8 || isempty(maxit),maxit=1000;end
if nargin<7 || isempty(m),m=5;end
if nargin<6 || isempty(delta),delta=1e-4;end
if nargin<5 || isempty(gamma),gamma=[];end
if m<1 || m~=floor(m) || ~isscalar(c) || isnan(c) || c<=0, error('Invalid m or c');end
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
x_hist=zeros(n,maxit+1);x_hist(:,1)=x;
acc=0;reject_objective=0;reject_consistency=0;
accept_hist=false(1,maxit);consistency_hist=nan(1,maxit);bound_hist=nan(1,maxit);
yk=x-gamma*grad(x);x1=prox(yk);phi=obj(x1);
F=x1-gamma*grad(x1)-yk;
xprev=x;x=x1;k_final=1;x_hist(:,2)=x;
f_hist(2)=phi;t_hist(2)=toc(clock);res_hist(1)=abs(phi-phik)/(1+abs(phik));phik=phi;
Ybuf=zeros(n,m);Hbuf=zeros(n,m);count=0;
for k=2:maxit
    zk=yk+F; % EXACT identity only for the exact-residual variant;
              % for the inexact variant compute the true z below.
    zk=x-gamma*grad(x); % Required for consistent safeguard/fallback.
    yt=yk+F;xt=prox(yt);Ft=xt-gamma*grad(xt)-yt;
    if count<m,count=count+1;else,Ybuf(:,1:end-1)=Ybuf(:,2:end);Hbuf(:,1:end-1)=Hbuf(:,2:end);end
    Ybuf(:,count)=yk;Hbuf(:,count)=F;
    Y=[Ybuf(:,1:count),yt];H=[Hbuf(:,1:count),Ft];s=size(H,2);
    C=H'*H;C=(C+C')/2;v=(C+1e-12*eye(s))\ones(s,1);
    valid=all(isfinite(v)) && isfinite(sum(v)) && abs(sum(v))>eps;
    if valid
        alpha=v/sum(v);yc=Y*alpha;xc=prox(yc);
        phi_candidate=obj(xc);
        pass_obj=isfinite(phi_candidate) && phi_candidate<=phik+ (1/k^2-delta)*norm(xc-x)^2;
        err=norm(yc-zk);bound=c*norm(x-xprev);
        pass_consistency=isinf(c) || (isfinite(err) && err<=bound);
    else
        pass_obj=false;pass_consistency=false;err=NaN;bound=c*norm(x-xprev);
    end
    consistency_hist(k)=err;bound_hist(k)=bound;
    if ~pass_obj,reject_objective=reject_objective+1;
    elseif ~pass_consistency,reject_consistency=reject_consistency+1;end
    if pass_obj && pass_consistency
        xnew=xc;ynew=yc;phi=phi_candidate;acc=acc+1;accept_hist(k)=true;
        Fnew=H*alpha; % Inexact residual update is intentional.
    else
        ynew=zk;xnew=prox(zk);phi=obj(xnew);
        Fnew=xnew-gamma*grad(xnew)-ynew;
    end
    res=abs(phi-phik)/(1+abs(phik));xprev=x;x=xnew;yk=ynew;F=Fnew;phik=phi;k_final=k;
    x_hist(:,k+1)=x;f_hist(k+1)=phi;res_hist(k)=res;t_hist(k+1)=toc(clock);
    if res<tol,break;end
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
info.x_hist=x_hist(:,1:k_final+1);info.c=c;
info.accept_hist=accept_hist(1:k_final);info.accept_rate=acc/max(k_final-1,1);
info.reject_objective=reject_objective;info.reject_consistency=reject_consistency;
info.consistency_hist=consistency_hist(1:k_final);info.bound_hist=bound_hist(1:k_final);
end
