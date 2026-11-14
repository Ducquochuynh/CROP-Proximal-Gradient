function [x,info] = student_solver_optimized(p,x0,opts)
% PG, FISTA, AA-PG, CROP-(in)exact and AdaPG updates are adapted from
% the original logistic L1 files; only f and grad have changed.
% Stopping: relative successive objective change as in original scripts.
    % Consistency guard only acts on CROP candidates; Inf recovers baseline.
    if ~isfield(opts,'csafe') || isempty(opts.csafe), opts.csafe=1; end
    csafe=opts.csafe;
    assert(isscalar(csafe) && isreal(csafe) && (isinf(csafe) || ...
           (isfinite(csafe) && csafe>0)), 'csafe must be positive or Inf');
    if ~isfield(opts,'print_every'),opts.print_every=25;end
    gamma=opts.gamma;
    maxit=opts.maxit;
    tol=opts.tol;
    memory=opts.memory;
    typ=opts.method;
    if maxit<1, error('maxit must be >=1.'); end
    if strcmp(typ,'AdaPG')
        if ~isfield(opts,'adapg_q') || isempty(opts.adapg_q)
            q_adapg=1.5;
        else
            q_adapg=opts.adapg_q;
        end
        assert(isscalar(q_adapg) && isfinite(q_adapg) && ...
            q_adapg>=1 && q_adapg<=2,'AdaPG requires q in [1,2].');
        gamma_previous=gamma;  % gamma_{-1}
        gamma_current=gamma;   % gamma_0, shared first PG step
        gamma_history=zeros(maxit,1);
        gamma_history(1)=gamma_current;
    end

    fh=zeros(maxit,1);
    th=zeros(maxit,1);
    rh=zeros(maxit,1);
    nacc=0;
    reject_objective=0; reject_consistency=0;
    accepted_hist=false(maxit,1);
    consistency_hist=nan(maxit,1);
    bound_hist=nan(maxit,1);
    cpu0=cputime;
    t0=tic;
    x=x0(:);
    phi_old=p.phi(x);
    grad_old=p.grad(x);
    y=x;                  % auxiliary point (for AA/CROP), FISTA momentum
    tf=1;                 % FISTA momentum parameter
    % Fixed-capacity buffers avoid reallocating very wide E2006 vectors.
    Rbuf=[]; Gbuf=[]; aa_count=0;
    Ybuf=[]; Hbuf=[]; crop_count=0;
    if strcmp(typ,'AA-PG')
        Rbuf=zeros(p.n,memory+1); Gbuf=zeros(p.n,memory+1);
    elseif strncmp(typ,'CROP-',5)
        Ybuf=zeros(p.n,memory); Hbuf=zeros(p.n,memory);
    end
    F=zeros(p.n,1);
    stop_met=false;

    % First standard PG step shared by all nine methods.
    % For AdaPG: x0 supplied by caller is x^{-1}; xnew is x^{0}.
    if strcmp(typ,'AdaPG')
        adapg_xprevious=x;
        adapg_gprevious=grad_old;
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
        Rbuf(:,1)=z0-x;
        Gbuf(:,1)=z0;
        aa_count=1;
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
    previous_iterate=x;
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
                % Secant curvature at (x^{k-1},x^{k-2}).
                grad_current=p.grad(old_x);
                sk=old_x-adapg_xprevious;
                yk=grad_current-adapg_gprevious;
                sn2=sk'*sk;
                if sn2>0 && isfinite(sn2)
                    ellk=(sk'*yk)/sn2;
                    Lk=norm(yk)/sqrt(sn2);
                else
                    % No informative secant: the curvature bound vanishes.
                    ellk=0;
                    Lk=0;
                end
                curvature=gamma_current^2*Lk^2 ...
                    -(2-q_adapg)*gamma_current*ellk+1-q_adapg;
                grow=sqrt(1/q_adapg+gamma_current/gamma_previous);
                if max(curvature,0)==0
                    bound=inf; % screenshot convention 1/0=Inf
                else
                    bound=1/sqrt(2*max(curvature,0));
                end
                gamma_next=gamma_current*min(grow,bound);
                if ~isfinite(gamma_next) || gamma_next<=0
                    error('AdaPG:InvalidStep','AdaPG stepsize became invalid at iteration %d.',k);
                end
                xnew=p.prox(old_x-gamma_next*grad_current, ...
                    gamma_next*p.lambda);
                adapg_xprevious=old_x;
                adapg_gprevious=grad_current;
                gamma_previous=gamma_current;
                gamma_current=gamma_next;
                gamma_history(k)=gamma_next;

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
                aa_count=min(aa_count+1,memory+1);
                if aa_count==memory+1 && k>memory+1
                    Rbuf(:,1:memory)=Rbuf(:,2:memory+1);
                    Gbuf(:,1:memory)=Gbuf(:,2:memory+1);
                end
                Rbuf(:,aa_count)=rk;
                Gbuf(:,aa_count)=gk;
                [alpha,ok]=affine_coeff(Rbuf(:,1:aa_count),opts.eps_reg);
                if ok
                    yaa=Gbuf(:,1:aa_count)*alpha;
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
                crop_count=min(crop_count+1,memory);
                if crop_count==memory && k>memory+1
                    Ybuf(:,1:memory-1)=Ybuf(:,2:memory);
                    Hbuf(:,1:memory-1)=Hbuf(:,2:memory);
                end
                Ybuf(:,crop_count)=y;
                Hbuf(:,crop_count)=F;
                Yk=[Ybuf(:,1:crop_count),ytilde];
                Hk=[Hbuf(:,1:crop_count),Ftilde];
                [alpha,ok]=affine_coeff(Hk,opts.eps_reg);
                if ok
                    ycr=Yk*alpha;
                    xcr=p.prox(ycr,gamma*p.lambda);
                    fcr=p.phi(xcr);
                    % Original safeguard: eps_k=1/(k_iter+1)^2, with
                    % k_iter=k-1 so eps_k=1/k^2 (first trial at k=2).
                    eps_k=1/k^2;
                    objective_ok=isfinite(fcr) && ...
                        fcr <= phi_old+(eps_k-opts.delta)*norm(xcr-old_x)^2;
                    % For the inexact variant, y+F is NOT necessarily z_k.
                    if isinf(csafe)
                        consistency_ok=true;
                        bound_hist(k)=Inf;
                    else
                        if strcmp(typ,'CROP-Exact')
                            z_current=old_x-gamma*grad_old;
                        else
                            z_current=old_x-gamma*p.grad(old_x);
                        end
                        consistency_hist(k)=norm(ycr-z_current);
                        bound_hist(k)=csafe*norm(old_x-previous_iterate);
                        consistency_ok=isfinite(consistency_hist(k)) && ...
                            consistency_hist(k)<=bound_hist(k);
                    end
                    if ~objective_ok, reject_objective=reject_objective+1; end
                    if objective_ok && ~consistency_ok
                        reject_consistency=reject_consistency+1;
                    end
                    if objective_ok && consistency_ok
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
                        if ~isinf(csafe) && exist('z_current','var')
                            ynew=z_current;
                        else
                            ynew=old_x-gamma*p.grad(old_x);
                        end
                    end
                    if strcmp(typ,'CROP-Exact')
                        ynew=old_x-gamma*grad_current;
                    end
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
        accepted_hist(k)=accepted;
        rh(k)=abs(pnew-phi_old)/(1+abs(phi_old));
        fh(k)=pnew;
        th(k)=toc(t0);
        iter_done=k;
        previous_iterate=old_x;
        x=xnew;
        phi_old=pnew;
        stop_met=(rh(k)<tol);
        report_iter(typ,k,pnew,rh(k),accepted,opts.print_every);
    end

    info.elapsed_time=toc(t0);
    info.cpu_time=cputime-cpu0;
    info.c=csafe;
    info.accept_hist=accepted_hist(1:iter_done);
    info.accept_rate=nacc/max(1,iter_done-1);
    info.consistency_hist=consistency_hist(1:iter_done);
    info.bound_hist=bound_hist(1:iter_done);
    info.reject_objective=reject_objective;
    info.reject_consistency=reject_consistency;
    info.f_hist=fh(1:iter_done);
    info.t_hist=th(1:iter_done);
    info.res_hist=rh(1:iter_done);
    info.k=iter_done;
    info.final_obj=fh(iter_done);
    info.stop_met=stop_met;
    info.n_accepted=nacc;
    if strcmp(typ,'AdaPG')
        info.gamma_hist=gamma_history(1:iter_done);
        info.adapg_q=q_adapg;
    end
end
