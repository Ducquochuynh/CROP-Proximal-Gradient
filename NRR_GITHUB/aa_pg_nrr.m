function [x,info,acc] = aa_pg_nrr(A,b,lambda,x0,gamma,m,maxit,tol,opts)
% Optimized NRR solver: min_{x>=0} 0.5*||A*x-b||^2 + lambda/2*||x||^2.
% info.k = total projected-gradient/proximal updates, including initialization.
% info.t_hist = elapsed solver time at each recorded objective.
% info.elapsed_time = total wall time including setup; info.cpu_time = process CPU seconds.
if nargin<9 || isempty(opts),opts=struct();end
if ~isfield(opts,'reg_eps'), opts.reg_eps=1e-12; end
if ~isfield(opts,'guard'), opts.guard=1e-4; end
if nargin<8 || isempty(tol),tol=1e-8;end
if nargin<7 || isempty(maxit),maxit=1000;end
if nargin<6 || isempty(m),m=5;end
if nargin<5,gamma=[];end
[rows,n]=size(A); b=b(:);
if numel(b)~=rows,error('Dimension mismatch');end
if nargin<4 || isempty(x0),x0=zeros(n,1);end
x0=x0(:);
if isempty(gamma),gamma=1/(normest(A)^2+lambda);end
if m<1 || maxit<1 || lambda<0,error('Invalid arguments');end
cpu0=cputime; wall0=tic;
[grad,obj,backend]=nrr_operators(A,b,lambda);
setup_time=toc(wall0); solver_start=tic;
f_hist=zeros(1,maxit+1); t_hist=zeros(1,maxit+1); res_hist=zeros(1,maxit);
accept_hist=false(1,maxit); f_hist(1)=obj(x0);
y=x0-gamma*grad(x0); x=max(y,0); phi=obj(x);
f_hist(2)=phi;t_hist(2)=toc(solver_start);
res_hist(1)=abs(phi-f_hist(1))/(1+abs(f_hist(1)));
R=zeros(n,0);G=zeros(n,0);acc=0;k=1;
if res_hist(1)>=tol
for k=2:maxit
    gx=grad(x); g=x-gamma*gx; r=g-y;
    R=[R,r]; G=[G,g];
    if size(R,2)>m+1,R=R(:,end-m:end);G=G(:,end-m:end);end
    H=R'*R; s=size(H,1);H=(H+H')/2;
    w=(H+opts.reg_eps*eye(s))\ones(s,1);
    if ~all(isfinite(w)) || ~isfinite(sum(w)) || sum(w)==0
        alpha=zeros(s,1);alpha(end)=1;
    else, alpha=w/sum(w); end
    ytry=G*alpha; xtry=max(ytry,0); ftry=obj(xtry);
    % Preserve the original AA-PG NRR full-objective safeguard.
    pass=isfinite(ftry) && ftry<=phi-opts.guard*(gamma/2)*sum(gx.^2);
    if pass,xnew=xtry;ynew=ytry;phinew=ftry;acc=acc+1;
    else,ynew=g;xnew=max(g,0);phinew=obj(xnew);end
    accept_hist(k)=pass;
    res_hist(k)=abs(phinew-phi)/(1+abs(phi));
    f_hist(k+1)=phinew;t_hist(k+1)=toc(solver_start);
    x=xnew;y=ynew;phi=phinew;
    if res_hist(k)<tol,break;end
end
end
info.k=k;info.f_hist=f_hist(1:k+1);info.t_hist=t_hist(1:k+1);
info.res_hist=res_hist(1:k);info.accept_hist=accept_hist(1:k);
info.accept_rate=acc/max(k-1,1);info.elapsed_time=toc(wall0);
info.cpu_time=cputime-cpu0;info.setup_time=setup_time;
info.backend=backend;info.final_objective=phi;
fprintf('AA-PG (m=%d)      iterations=%5d  time=%.4fs  accepted=%d  objective=%.9e\n',m,k,info.elapsed_time,acc,phi);
end
