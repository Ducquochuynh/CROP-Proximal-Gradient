function [x,info] = restoration_solvers(p,x0,opts)
% The PG, FISTA, AA-PG, CROP-(in)exact updates are adapted directly from
% the original logistic L1 files; only f and grad have changed.
% Stopping: relative successive objective change as in original scripts.
% AdaPG follows AdaPG^{q,q/2} with local L_k and ell_k.
% Its initial PG update is x^{-1}->x^0; subsequent steps are adaptive.
    gamma=opts.gamma;
    maxit=opts.maxit;
    tol=opts.tol;
    memory=opts.memory;
    typ=opts.method;
    if maxit<1, error('maxit must be >=1.'); end
    is_adapg=strcmp(typ,'AdaPG');
    if is_adapg
        q_adapg=1.5;
        if isfield(opts,'adapg_q') && ~isempty(opts.adapg_q)
            q_adapg=opts.adapg_q;
        end
        if ~isscalar(q_adapg) || ~isfinite(q_adapg) || q_adapg<1 || q_adapg>2
            error('AdaPG parameter q must belong to [1,2].');
        end
        if ~(isfinite(gamma) && gamma>0)
            error('AdaPG requires a strictly positive finite initial stepsize.');
        end
        gamma_previous=gamma;  % gamma_{-1}
        gamma_current=gamma;   % gamma_0
        gamma_hist=zeros(maxit,1);
    end

    fh=zeros(maxit,1);
    th=zeros(maxit,1);
    rh=zeros(maxit,1);
    nacc=0;
    t0=tic;
    x=x0(:);
    phi_old=p.phi(x);
    grad_old=p.grad(x);
    y=x;                  % auxiliary point (for AA/CROP), FISTA momentum
    tf=1;                 % FISTA momentum parameter
    Rbuf=zeros(p.n,0); Gbuf=zeros(p.n,0);
    Ybuf=zeros(p.n,0); Hbuf=zeros(p.n,0);
    F=zeros(p.n,1);
    stop_met=false;

    % First standard PG step; for AdaPG this is x^(-1) -> x^(0).
    if is_adapg
        adapg_xprev=x;         % x^(-1)
        adapg_gprev=grad_old;  % grad f(x^(-1))
        gamma_hist(1)=gamma_current;
    end
    z0=x-gamma*grad_old;
    xnew=p.prox(z0,gamma*p.lambda);
    phi_new=p.phi(xnew);
    if ~isfinite(phi_new), error('Nonfinite objective in first step.'); end
    k=1;
    iter_done=1;
    fh(k)=phi_new;
    rh(k)=abs(phi_new-phi_old)/(1+abs(phi_old));
    if strcmp(typ,'AA-PG')
        Rbuf=z0-x;
        Gbuf=z0;
        y=z0;
    elseif strncmp(typ,'CROP-',5)
        y=z0;
        grad_c=p.grad(xnew); % required for initial CROP residual
        F=xnew-gamma*grad_c-y;
        if strcmp(typ,'CROP-Exact'), grad_old=grad_c; end
    elseif strcmp(typ,'FISTA')
        tfnew=(1+sqrt(1+4*tf^2))/2;
        y=xnew+((tf-1)/tfnew)*(xnew-x);
        tf=tfnew;
    end
    x=xnew;
    phi_old=phi_new;
    th(k)=toc(t0);
    stop_met=(rh(k)<tol);
    report_iter(typ,k,phi_new,rh(k),false,opts.print_every);
    % For a common initialization, the first PG step is still subject to
    % the requested stopping rule (unlike an extra uncounted warm-up).

    for k=2:maxit
        if stop_met, break; end
        old_x=x;
        accepted=false;
        pnew=NaN;
        ynew=[];
        Fnew=[];

        switch typ
            case 'AdaPG'
                % At this point old_x=x^k, adapg_xprev=x^(k-1).
                % For k=2 in MATLAB: old_x=x^0 and adapg_xprev=x^(-1).
                grad_current=p.grad(old_x);
                dx=old_x-adapg_xprev;
                dg=grad_current-adapg_gprev;
                dx2=dx'*dx;
                if dx2>0
                    Lloc=norm(dg)/sqrt(dx2);
                    ellloc=(dx'*dg)/dx2;
                else
                    Lloc=0;
                    ellloc=0;
                end
                growth=sqrt(1/q_adapg+gamma_current/gamma_previous);
                D=gamma_current^2*Lloc^2 ...
                    -(2-q_adapg)*gamma_current*ellloc+1-q_adapg;
                if D>0
                    curvature=1/sqrt(2*D);
                else
                    curvature=inf;  % convention 1/0 = +infinity
                end
                gamma_next=gamma_current*min(growth,curvature);
                if ~isfinite(gamma_next) || gamma_next<=0
                    error('AdaPG produced an invalid stepsize at iteration %d.',k);
                end
                xnew=p.prox(old_x-gamma_next*grad_current, ...
                            gamma_next*p.lambda);
                adapg_xprev=old_x;
                adapg_gprev=grad_current;
                gamma_previous=gamma_current;
                gamma_current=gamma_next;
                gamma_hist(k)=gamma_next;

            case 'PG' 
                grad_current=p.grad(old_x);
                zpg=old_x-gamma*grad_current;
                xnew=p.prox(zpg,gamma*p.lambda);

            case 'FISTA'
                gy=p.grad(y);
                xnew=p.prox(y-gamma*gy,gamma*p.lambda);
                tfnew=(1+sqrt(1+4*tf^2))/2;
                ynew=xnew+((tf-1)/tfnew)*(xnew-old_x);
                tf=tfnew;

            case 'AA-PG'
                grad_current=p.grad(old_x);
                zpg=old_x-gamma*grad_current;
                gk=zpg;
                rk=gk-y;
                Rbuf=[Rbuf,rk]; %#ok<AGROW>
                Gbuf=[Gbuf,gk]; %#ok<AGROW>
                if size(Rbuf,2)>memory+1
                    Rbuf=Rbuf(:,end-memory:end);
                    Gbuf=Gbuf(:,end-memory:end);
                end
                [alpha,ok]=affine_coeff(Rbuf,opts.eps_reg);
                if ok
                    yaa=Gbuf*alpha;
                    xaa=p.prox(yaa,gamma*p.lambda);
                    f_aa=p.f(xaa);
                    % Keep original AA-PG *smooth-only* acceptance test.
                    if isfinite(f_aa) && ...
                       f_aa <= p.f(old_x)-opts.theta_aa*gamma/2*sum(grad_current.^2)
                        xnew=xaa; ynew=yaa; accepted=true;
                        pnew=f_aa+p.h(xnew); nacc=nacc+1;
                    end
                end
                if ~accepted
                    ynew=zpg;
                    xnew=p.prox(ynew,gamma*p.lambda);
                end

            case {'CROP-Inexact','CROP-Exact'}
                ytilde=y+F;
                xtilde=p.prox(ytilde,gamma*p.lambda);
                Ftilde=xtilde-gamma*p.grad(xtilde)-ytilde;
                Ybuf=[Ybuf,y]; %#ok<AGROW>
                Hbuf=[Hbuf,F]; %#ok<AGROW>
                if size(Ybuf,2)>memory
                    Ybuf=Ybuf(:,end-memory+1:end);
                    Hbuf=Hbuf(:,end-memory+1:end);
                end
                Yk=[Ybuf,ytilde];
                Hk=[Hbuf,Ftilde];
                [alpha,ok]=affine_coeff(Hk,opts.eps_reg);
                if ok
                    ycr=Yk*alpha;
                    xcr=p.prox(ycr,gamma*p.lambda);
                    fcr=p.phi(xcr);
                    % Original safeguard: eps_k=1/(k_iter+1)^2, with
                    % k_iter=k-1 so eps_k=1/k^2 (first trial at k=2).
                    eps_k=1/k^2;
                    if isfinite(fcr) && ...
                       fcr <= phi_old+(eps_k-opts.delta)*norm(xcr-old_x)^2
                        xnew=xcr; ynew=ycr; pnew=fcr;
                        accepted=true; nacc=nacc+1;
                        if strcmp(typ,'CROP-Inexact')
                            Fnew=Hk*alpha;
                        end
                    end
                end
                if ~accepted
                    % CROP-Inexact needs grad(old_x) ONLY on fallback.
                    % CROP-Exact can reuse the exact gradient from F(old_x).
                    if strcmp(typ,'CROP-Exact')
                        grad_current=grad_old;
                    else
                        grad_current=p.grad(old_x);
                    end
                    ynew=old_x-gamma*grad_current;
                    xnew=p.prox(ynew,gamma*p.lambda);
                end

            otherwise
                error('Unknown method: %s',typ);
        end

        if ~isfinite(pnew), pnew=p.phi(xnew); end
        if ~isfinite(pnew) || any(~isfinite(xnew))
            error('%s produced a nonfinite iterate/objective at k=%d.',typ,k);
        end
        if strncmp(typ,'CROP-',5)
            if strcmp(typ,'CROP-Exact')
                % Exact residual variant always evaluates true F(xnew).
                grad_old=p.grad(xnew);
                Fnew=xnew-gamma*grad_old-ynew;
            elseif isempty(Fnew)
                % Inexact residual needs an exact residual only on fallback.
                Fnew=xnew-gamma*p.grad(xnew)-ynew;
            end
            F=Fnew;
        end
        if ~isempty(ynew), y=ynew; end
        rh(k)=abs(pnew-phi_old)/(1+abs(phi_old));
        fh(k)=pnew;
        th(k)=toc(t0);
        iter_done=k;
        x=xnew;
        phi_old=pnew;
        stop_met=(rh(k)<tol);
        report_iter(typ,k,pnew,rh(k),accepted,opts.print_every);
    end

    info.f_hist=fh(1:iter_done);
    info.t_hist=th(1:iter_done);
    info.res_hist=rh(1:iter_done);
    info.k=iter_done;
    info.final_obj=fh(iter_done);
    info.stop_met=stop_met;
    info.n_accepted=nacc;
    if is_adapg
        info.gamma_hist=gamma_hist(1:iter_done);
        info.adapg_q=q_adapg;
    end
end


function [alpha,ok]=affine_coeff(F,reg)
    % Solve min ||F*alpha||^2 + reg||alpha||^2, s.t. sum(alpha)=1.
    p=size(F,2);
    ok=false; alpha=zeros(p,1);
    if p<1 || any(~isfinite(F(:))),return;end
    gram=F'*F;
    regscale=max(1,norm(gram,'fro'));
    gram=gram+max(reg,eps)*regscale*eye(p);
    v=gram\ones(p,1);
    d=sum(v);
    if isfinite(d) && abs(d)>eps && all(isfinite(v))
        alpha=v/d;
        ok=all(isfinite(alpha));
    end
end

function report_iter(name,k,f,res,accepted,period)
    if period>0 && (k==1 || mod(k,period)==0)
        fprintf('%s: iter=%d, obj=%.12e, relchange=%.3e, accepted=%d\n', ...
            name,k,f,res,accepted);
    end
end
