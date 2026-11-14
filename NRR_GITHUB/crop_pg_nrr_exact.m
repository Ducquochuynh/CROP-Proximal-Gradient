function [x,info,acc] = crop_pg_nrr_exact(A,b,lambda,x0,gamma,delta,m,maxit,tol,c)
% Optimized NRR solver: min_{x>=0} 0.5*||A*x-b||^2 + lambda/2*||x||^2.
% info.k = total projected-gradient/proximal updates, including initialization.
% info.t_hist = elapsed solver time at each recorded objective.
% info.elapsed_time = total wall time including setup; info.cpu_time = process CPU seconds.
% c=Inf disables consistency safeguard; finite c enforces
% ||y_Cr - (x_k-gamma*grad(x_k))|| <= c*||x_k-x_{k-1}||.
if nargin<10 || isempty(c), c=1;end
if ~(isscalar(c) && isreal(c) && c>0 && ~isnan(c)),error('Invalid c');end
if nargin<9 || isempty(tol),tol=1e-8;end
if nargin<8 || isempty(maxit),maxit=1000;end
if nargin<7 || isempty(m),m=5;end
if nargin<6 || isempty(delta),delta=1e-4;end
if nargin<5,gamma=[];end
[rows,n]=size(A);b=b(:);
if numel(b)~=rows,error('Dimension mismatch');end
if nargin<4 || isempty(x0),x0=zeros(n,1);end
x0=x0(:);
if isempty(gamma),gamma=1/(normest(A)^2+lambda);end
if m<1 || maxit<1 || lambda<0,error('Invalid inputs');end
cpu0=cputime;wall0=tic;
[grad,obj,backend]=nrr_operators(A,b,lambda);
setup_time=toc(wall0);solver_start=tic;
f_hist=zeros(1,maxit+1);t_hist=zeros(1,maxit+1);res_hist=zeros(1,maxit);
accept_hist=false(1,maxit);consistency_hist=nan(1,maxit);
bound_hist=nan(1,maxit);
xprev=x0; phi0=obj(xprev);f_hist(1)=phi0;
y=xprev-gamma*grad(xprev);x=max(y,0);phi=obj(x);
F=x-gamma*grad(x)-y;
f_hist(2)=phi;t_hist(2)=toc(solver_start);
res_hist(1)=abs(phi-phi0)/(1+abs(phi0));
Y=zeros(n,0);H=zeros(n,0);acc=0;reject_obj=0;reject_consistency=0;k=1;
if res_hist(1)>=tol
for k=2:maxit
    % The current exact residual satisfies y+F=z_k for the exact version.
    zk=x-gamma*grad(x); tilde_y=zk;
    xtilde=max(tilde_y,0);
    Ft=xtilde-gamma*grad(xtilde)-tilde_y;
    Y=[Y,y];H=[H,F];
    if size(Y,2)>m,Y=Y(:,end-m+1:end);H=H(:,end-m+1:end);end
    Yk=[Y,tilde_y];Hk=[H,Ft];
    E=Hk'*Hk; E=(E+E')/2; s=size(E,1);
    reg=1e-12; w=(E+reg*eye(s))\ones(s,1);
    if ~all(isfinite(w)) || ~isfinite(sum(w)) || sum(w)==0
       alpha=zeros(s,1);alpha(end)=1;
    else,alpha=w/sum(w);end
    ytry=Yk*alpha;xtry=max(ytry,0);ftry=obj(xtry);
    epsk=1/(k+1)^2;
    passobj=isfinite(ftry) && ftry<=phi+(epsk-delta)*sum((xtry-x).^2);
    cerr=norm(ytry-zk); cbound=c*norm(x-xprev);
    passcons=isinf(c) || (isfinite(cerr) && cerr<=cbound);
    consistency_hist(k)=cerr;bound_hist(k)=cbound;
    if ~passobj,reject_obj=reject_obj+1;end
    if passobj && ~passcons,reject_consistency=reject_consistency+1;end
    accept=passobj && passcons;
    if accept
      xnew=xtry;ynew=ytry;phinew=ftry;acc=acc+1;
      Fnew=xnew-gamma*grad(xnew)-ynew;
    else
      ynew=zk;xnew=max(zk,0);phinew=obj(xnew);
      Fnew=xnew-gamma*grad(xnew)-ynew;
    end
    accept_hist(k)=accept;
    res_hist(k)=abs(phinew-phi)/(1+abs(phi));
    f_hist(k+1)=phinew;t_hist(k+1)=toc(solver_start);
    xprev=x;x=xnew;y=ynew;F=Fnew;phi=phinew;
    if res_hist(k)<tol,break;end
end
end
info.k=k;info.f_hist=f_hist(1:k+1);info.t_hist=t_hist(1:k+1);
info.res_hist=res_hist(1:k);info.accept_hist=accept_hist(1:k);
info.consistency_hist=consistency_hist(1:k);info.bound_hist=bound_hist(1:k);
info.reject_objective=reject_obj;info.reject_consistency=reject_consistency;
info.accept_rate=acc/max(k-1,1);info.c=c;
info.elapsed_time=toc(wall0);info.cpu_time=cputime-cpu0;
info.setup_time=setup_time;info.backend=backend;info.final_objective=phi;
fprintf('CROP exact');fprintf(' (m=%d,c=%g) iterations=%5d  time=%.4fs  accepted=%d  objective=%.9e\n',m,c,k,info.elapsed_time,acc,phi);
end
