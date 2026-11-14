%% Convex sparse image restoration: original CROP-PG algorithms and figures
% MATLAB R2024b; no Wavelet Toolbox or Image Processing Toolbox required.
% Ten methods: 9 original competitors plus PGM-pc Algorithm 5.1.
% Requires pgmpc_restoration_optimized.m alongside original solver.
% min_x 0.5*||A*W'*x-b||^2 + lambda*||x||_1
% W = orthonormal multilevel 2D Haar, A = periodic Gaussian blur.
% Just press Run: Set12 image is downloaded automatically on the first run.
% Requires Internet on first Set12 download; subsequent runs are offline.
clearvars; clc; close all;
rng(0,'twister');
root=fileparts(mfilename('fullpath'));
if isempty(root),root=pwd;end
addpath(root);

%% User settings
cfg.data_dir=fullfile(root,'data');
cfg.dataset='Set12';           % 'Set12', 'BSDS500', 'folder', or 'file'
cfg.folder_dir=fullfile(root,'data','my_images'); % for dataset='folder'
cfg.output_dir=fullfile(root,'results_Set12_Batch_10Methods');
cfg.image_file='';              % for dataset='file': full path to JPG/PNG/TIF/BMP
cfg.image_index=1;             % Set12 index 1..12, or sorted image index
cfg.auto_download=true;        % Set12 auto-download or BSDS500 auto-download
cfg.image_size=256;           % 128, 256, 512; 512 may upscale smaller source images
cfg.wavelet_levels=4;
cfg.blur_sigma=3.0;           % Gaussian blur std in pixels
cfg.noise_sigma=0.03;         % noise as fraction of [0,1] intensity range
cfg.lambda_fraction=1e-3;    % lambda = fraction * ||B'*b||_inf
cfg.norm_safety=1.05;
cfg.memories=[10 20];
cfg.maxit=2000;               % maximum iterations for each method
cfg.tol=1e-13;               % identical relative-objective-change stopping
cfg.delta=1e-4;
cfg.theta_aa=1e-4;
cfg.eps_reg=1e-12;
cfg.print_every=0;
cfg.pgmpc_alpha=0.99;  % Algorithm 5.1
cfg.adapg_q=1.5;          % AdaPG parameter q in [1,2]
cfg.save_figures=true; % legacy setting
cfg.save_per_image_figures=false; % true: 4 plots per image (48 figures)
cfg.save_per_image_mat=false; % true: large MAT files with solutions/history
cfg.save_progress=true;   % save accumulating CSV after each image
cfg.random_seed=0;
cfg.image_indices=1:12; % default: all 12; use 1:2 for a quick test
cfg.demo_mode=false;         % true => MATLAB-generated synthetic phantom
cfg.resize_mode='preserve_crop'; % 'preserve_crop' or 'stretch'; no extra toolbox

if ~exist(cfg.output_dir,'dir'),mkdir(cfg.output_dir);end
N=cfg.image_size;
assert(N>=16 && abs(log2(N)-round(log2(N)))<1e-12, ...
    'image_size must be a power of two >=16');
assert(cfg.wavelet_levels>=1 && cfg.wavelet_levels<=log2(N), ...
    'Invalid wavelet_levels');
assert(numel(cfg.memories)==2 && all(cfg.memories>=1), ...
    'Specify exactly two positive memory values');

% Batch experiments: require Set12 images, use the same solver settings.
cfg.dataset='Set12';
image_indices=cfg.image_indices;
assert(all(ismember(image_indices,1:12)) && all(image_indices==floor(image_indices)) ...
 && numel(unique(image_indices))==numel(image_indices), ...
 'cfg.image_indices must contain distinct integer indices between 1 and 12.');
all_rows=table();
mean_objectives=nan(numel(image_indices),1);
batch_clock=tic;
for image_pos=1:numel(image_indices)
%% Image preparation (outside solver timers)
image_id=image_indices(image_pos);
cfg.image_index=image_id;
image_outdir=fullfile(cfg.output_dir,sprintf('Set12_%02d',image_id));
if ~exist(image_outdir,'dir'), mkdir(image_outdir); end
% Deterministic degradation per image, independent of order/restarts.
rng(cfg.random_seed+image_id,'twister');
fprintf('\n===== Set12 image %02d (%d of %d) =====\n',image_id,image_pos,numel(image_indices));
if cfg.demo_mode
    % Deterministic synthetic check; do NOT identify this as BSDS500.
    [xx,yy]=meshgrid(linspace(-1,1,N));
    original=0.20+0.55*double(xx.^2+yy.^2<0.62^2) ...
        +0.20*double(abs(xx+0.33)<0.12 & abs(yy-0.20)<0.38);
    original=min(max(original,0),1);
    source='SYNTHETIC PHANTOM (not BSDS500)';
else
    % Resolve image: real photographs from Set12, BSDS500, local file, folder.
    if ~isempty(cfg.image_file)
        imgpath=cfg.image_file;
    else
        switch lower(cfg.dataset)
            case 'set12'
                imgpath=get_set12_image(fullfile(cfg.data_dir,'Set12'), ...
                    cfg.image_index,cfg.auto_download);
            case 'bsds500'
                imgpath=get_bsds500_image(cfg.data_dir, ...
                    cfg.image_index,cfg.auto_download);
            case 'folder'
                imgpath=choose_image(cfg.folder_dir,cfg.image_index);
            case 'file'
                error('For dataset=file, set cfg.image_file to an existing image.');
            otherwise
                error('Unknown cfg.dataset: %s',cfg.dataset);
        end
    end
    if ~isfile(imgpath),error('Image not found: %s',imgpath);end
    raw=imread(imgpath);
    if size(raw,3)>3, raw=raw(:,:,1:3); end % Ignore alpha if present.
    if isinteger(raw)
        raw=double(raw)/double(intmax(class(raw)));
    else
        raw=double(raw);
        if max(raw(:))>1, warning('Floating-point image exceeds 1; scaling by its maximum.');raw=raw/max(raw(:));end
    end
    if size(raw,3)==3
        raw=0.2989*raw(:,:,1)+0.5870*raw(:,:,2)+0.1140*raw(:,:,3);
    end
    raw=min(max(raw,0),1);
    original=prepare_square_image(raw,N,cfg.resize_mode);
    source=imgpath;
    fprintf('Source resolution: %d x %d -> experimental resolution: %d x %d (%s).\n', ...
        size(raw,1),size(raw,2),N,N,cfg.resize_mode);
    if N>min(size(raw,1),size(raw,2))
        fprintf('NOTE: source image was enlarged; this does not add image information.\n');
    end
end

% Periodic blur. PSF indexed by circular distance from the origin.
q=(0:N-1); dist=min(q,N-q);
[dx,dy]=meshgrid(dist,dist);
psf=exp(-(dx.^2+dy.^2)/(2*cfg.blur_sigma^2));
psf=psf/sum(psf(:));
H=fft2(psf);
blur=@(u) real(ifft2(H.*fft2(u)));
blurT=@(u) real(ifft2(conj(H).*fft2(u)));
clean_blurred=blur(original);
observed=clean_blurred+cfg.noise_sigma*randn(N,N);
b=observed(:);

% Orthonormal Haar: W' maps coefficient vector to image.
W=@(v) haar2_orth(reshape(v,N,N),cfg.wavelet_levels,false);
WT=@(v) haar2_orth(reshape(v,N,N),cfg.wavelet_levels,true);
B=@(x) reshape(blur(WT(x)),[],1);
BT=@(r) reshape(W(blurT(reshape(r,N,N))),[],1);
n=N*N;

%% Convex composite objective, gradient, proximal operator
prob.n=n;
prob.f=@(x) 0.5*sum((B(x)-b).^2);
prob.grad=@(x) BT(B(x)-b);
prob.prox=@(z,t) sign(z).*max(abs(z)-t,0);
x0=zeros(n,1);
lambda_max=norm(BT(b),inf);
lambda=cfg.lambda_fraction*lambda_max;
prob.lambda=lambda;
prob.h=@(x) lambda*sum(abs(x));
prob.phi=@(x) prob.f(x)+prob.h(x);
% For orthonormal W, ||A W'||_2 = ||A||_2 <= 1 for normalized positive PSF.
% The DC frequency of the circular Gaussian blur has magnitude exactly 1.
spectral_norm=max(abs(H(:)));
L=(cfg.norm_safety*spectral_norm)^2;
gamma=1/L;
fprintf('Image: %s, size=%dx%d, levels=%d\n',source,N,N,cfg.wavelet_levels);
fprintf('lambda_max=%.6e, lambda=%.6e, L=%.6e, gamma=%.6e\n',lambda_max,lambda,L,gamma);
fprintf('Stop at objective relative change below %.2e\n',cfg.tol);

%% Run the original eight methods plus AdaPG and PGM-pc
labels={'PG','FISTA',sprintf('AA-PG (m = %d)',cfg.memories(1)), ...
 sprintf('AA-PG (m = %d)',cfg.memories(2)), ...
 sprintf('CROP-PG-Inexact (m = %d)',cfg.memories(1)), ...
 sprintf('CROP-PG-Inexact (m = %d)',cfg.memories(2)), ...
 sprintf('CROP-PG-Exact (m = %d)',cfg.memories(1)), ...
 sprintf('CROP-PG-Exact (m = %d)',cfg.memories(2)), ...
 'AdaPG','PGM-pc'};
methods={'PG','FISTA','AA-PG','AA-PG', ...
 'CROP-Inexact','CROP-Inexact','CROP-Exact','CROP-Exact','AdaPG','PGM-pc'};
mem=[0 0 cfg.memories cfg.memories cfg.memories 0 0];
nmethods=numel(methods);
infos=cell(nmethods,1); sols=cell(nmethods,1);
method_cpu=zeros(nmethods,1); method_wall=zeros(nmethods,1);
for ii=1:nmethods
 opts=struct('method',methods{ii},'memory',max(1,mem(ii)), ...
  'gamma',gamma,'maxit',cfg.maxit,'tol',cfg.tol,'delta',cfg.delta, ...
  'theta_aa',cfg.theta_aa,'eps_reg',cfg.eps_reg, ...
  'print_every',cfg.print_every,'adapg_q',cfg.adapg_q);
 opts.L=L; opts.alpha=cfg.pgmpc_alpha;
 fprintf('\n=========== %s ===========\n',labels{ii});
 cpu_start=cputime; wall_start=tic;
 if strcmp(methods{ii},'PGM-pc')
     [sols{ii},infos{ii}]=pgmpc(prob,x0,opts);
 else
     [sols{ii},infos{ii}]=restoration_solvers(prob,x0,opts);
 end
 method_wall(ii)=toc(wall_start);
 method_cpu(ii)=cputime-cpu_start;
end

%% Evaluation: common smallest FINAL objective (not a certified optimum)
final_values=cellfun(@(s) s.f_hist(end),infos);
[phi_ref,best_final_idx]=min(final_values);
fprintf('Common reference (best final objective): %.16e, method: %s\n', ...
    phi_ref,labels{best_final_idx});
Method=string(labels(:));
Iterations=zeros(nmethods,1); CPUTime_s=zeros(nmethods,1); WallTime_s=zeros(nmethods,1);
FinalObjective=zeros(nmethods,1); FinalRelChange=zeros(nmethods,1);
AcceptedPct=nan(nmethods,1); StopMet=false(nmethods,1);
PSNR_dB=zeros(nmethods,1); RelativeImageError=zeros(nmethods,1);
PGResidual=zeros(nmethods,1); ObjectiveGap=zeros(nmethods,1);
recon=cell(nmethods,1);
for ii=1:nmethods
 v=infos{ii};
 rec=WT(sols{ii}); recon{ii}=rec;
 Iterations(ii)=v.k; CPUTime_s(ii)=method_cpu(ii); WallTime_s(ii)=method_wall(ii);
 FinalObjective(ii)=v.final_obj;
 ObjectiveGap(ii)=v.final_obj-phi_ref;
 % Unit of residual: norm of gradient mapping / max(1,norm(x)).
 % This diagnostic does NOT change the original stopping rule.
 xx=sols{ii};
 prox_x=prob.prox(xx-gamma*prob.grad(xx),gamma*lambda);
 PGResidual(ii)=norm((xx-prox_x)/gamma)/max(1,norm(xx));
 FinalRelChange(ii)=v.res_hist(end);
 StopMet(ii)=v.stop_met;
 if strcmp(methods{ii},'AA-PG') || strncmp(methods{ii},'CROP-',5)
   AcceptedPct(ii)=100*v.n_accepted/max(1,v.k-1);
 end
 PSNR_dB(ii)=10*log10(1/mean((rec(:)-original(:)).^2));
 RelativeImageError(ii)=norm(rec(:)-original(:))/max(norm(original(:)),eps);
end
ImageIndex=repmat(image_id,nmethods,1);
ImageName=repmat(string(sprintf('%02d.png',image_id)),nmethods,1);
summary=table(ImageIndex,ImageName,Method,Iterations,CPUTime_s,WallTime_s, ...
 FinalObjective,ObjectiveGap,FinalRelChange,PGResidual, ...
 AcceptedPct,StopMet,PSNR_dB,RelativeImageError);
disp(summary(:,{'Method','Iterations','CPUTime_s','FinalObjective', ...
  'ObjectiveGap','PGResidual','PSNR_dB','StopMet'}));
all_rows=[all_rows;summary]; %#ok<AGROW>
mean_objectives(image_pos)=phi_ref;
fprintf('Observed degraded PSNR = %.3f dB\n', ...
 10*log10(1/mean((observed(:)-original(:)).^2)));

writetable(summary,fullfile(image_outdir,'ImageRestoration_summary.csv'));
if cfg.save_per_image_mat
 save(fullfile(image_outdir,'ImageRestoration_results.mat'), ...
  'cfg','source','lambda','L','gamma','infos','sols','labels','phi_ref','summary','-v7.3');
end
if cfg.save_per_image_figures
    plot_restoration_three(infos,labels,phi_ref,image_outdir,true);
end

% Show and export restored images, clearly labelled by method.
if cfg.save_per_image_figures
fig=figure('Name','Restored Images','Color','w','Position',[50 50 1200 800]);
for ii=1:3+nmethods
 subplot(ceil((3+nmethods)/4),4,ii);
 if ii==1, u=original; ttl='Original';
 elseif ii==2,u=clean_blurred;ttl='Blurred';
 elseif ii==3,u=observed;ttl='Observed (blur + noise)';
 else,jj=ii-3;u=recon{jj};ttl=sprintf('%s | %.2f dB',labels{jj},PSNR_dB(jj));end
 imagesc(u,[0 1]);axis image off;colormap(gray(256));
 title(ttl,'Interpreter','none','FontSize',10);
end
if cfg.save_figures
 exportgraphics(fig,fullfile(image_outdir,'ImageRestoration_Reconstructions.png'),'Resolution',200);
 exportgraphics(fig,fullfile(image_outdir,'ImageRestoration_Reconstructions.pdf'));
end
close(fig);
end
fprintf('Saved results for image %02d in %s\n',image_id,image_outdir);
if cfg.save_progress
    writetable(all_rows,fullfile(cfg.output_dir,'Set12_AllResults_PARTIAL.csv'));
end




end % for each Set12 image

%% Batch-level CSV files and manuscript-ready summary
writetable(all_rows,fullfile(cfg.output_dir,'Set12_AllResults.csv'));
methods_order=string(labels(:));
MeanIterations=zeros(nmethods,1); StdIterations=zeros(nmethods,1);
MeanCPUTime_s=zeros(nmethods,1); StdCPUTime_s=zeros(nmethods,1);
MeanPSNR_dB=zeros(nmethods,1); StdPSNR_dB=zeros(nmethods,1);
MeanPGResidual=zeros(nmethods,1); MedianPGResidual=zeros(nmethods,1);
MeanObjectiveGap=zeros(nmethods,1);
StoppedCount=zeros(nmethods,1); FastestCount=zeros(nmethods,1);
for j=1:nmethods
    rows=all_rows(all_rows.Method==methods_order(j),:);
    MeanIterations(j)=mean(rows.Iterations);
    StdIterations(j)=std(rows.Iterations);
    MeanCPUTime_s(j)=mean(rows.CPUTime_s);
    StdCPUTime_s(j)=std(rows.CPUTime_s);
    MeanPSNR_dB(j)=mean(rows.PSNR_dB);
    StdPSNR_dB(j)=std(rows.PSNR_dB);
    MeanPGResidual(j)=mean(rows.PGResidual);
    MedianPGResidual(j)=median(rows.PGResidual);
    MeanObjectiveGap(j)=mean(rows.ObjectiveGap);
    StoppedCount(j)=sum(rows.StopMet);
end
for i=1:numel(image_indices)
    rows=all_rows(all_rows.ImageIndex==image_indices(i),:);
    [~,winner]=min(rows.CPUTime_s);
    winner_name=rows.Method(winner);
    FastestCount(methods_order==winner_name)=FastestCount(methods_order==winner_name)+1;
end
Method=methods_order;
summary_batch=table(Method,MeanIterations,StdIterations,MeanCPUTime_s,...
 StdCPUTime_s,MeanPSNR_dB,StdPSNR_dB,MeanPGResidual,...
 MedianPGResidual,MeanObjectiveGap,StoppedCount,FastestCount);
disp('=== Average results over images ===');
disp(summary_batch);
writetable(summary_batch,fullfile(cfg.output_dir,'Set12_MethodSummary.csv'));

% Per-image x per-method matrices, convenient for LaTeX tables.
metrics={'Iterations','CPUTime_s','FinalObjective','PSNR_dB','PGResidual'};
for kk=1:numel(metrics)
    metric=metrics{kk};
    M=zeros(numel(image_indices),nmethods);
    for i=1:numel(image_indices)
       r=all_rows(all_rows.ImageIndex==image_indices(i),:);
       for j=1:nmethods
          M(i,j)=r.(metric)(r.Method==methods_order(j));
       end
    end
    colnames=matlab.lang.makeUniqueStrings(matlab.lang.makeValidName(cellstr(methods_order)));
    wide=array2table(M,'VariableNames',colnames);
    wide=addvars(wide,image_indices(:),'Before',1,'NewVariableNames','ImageIndex');
    writetable(wide,fullfile(cfg.output_dir,['Set12_',metric,'_Matrix.csv']));
end
save(fullfile(cfg.output_dir,'Set12_Batch_Summary.mat'), ...
 'cfg','all_rows','summary_batch','image_indices','mean_objectives');
fprintf('\nFINISHED %d Set12 images in %.1f seconds.\n',numel(image_indices),toc(batch_clock));
fprintf('Results saved in: %s\n',cfg.output_dir);

%% ===== Local helpers: no external MATLAB toolboxes needed =====
function filepath=get_set12_image(folder,index,allow_download)
    if index<1 || index>12 || index~=floor(index)
        error('Set12 image_index must be an integer from 1 to 12.');
    end
    if ~exist(folder,'dir'),mkdir(folder);end
    filepath=fullfile(folder,sprintf('%02d.png',index));
    if ~isfile(filepath)
        if ~allow_download
            error('Set12 %02d.png missing. Set cfg.auto_download=true or place the image in %s.',index,folder);
        end
        url=sprintf('https://raw.githubusercontent.com/cszn/DnCNN/master/testsets/Set12/%02d.png',index);
        fprintf('Downloading Set12 image %02d from cszn/DnCNN on GitHub...\n',index);
        try
            websave(filepath,url);
        catch ME
            error('Could not download Set12 image. Download manually from %s. Details: %s',url,ME.message);
        end
    end
end

function filepath=get_bsds500_image(data_dir,index,allow_download)
    folder=fullfile(data_dir,'BSR','BSDS500','data','images','test');
    if ~isfolder(folder) || isempty(dir(fullfile(folder,'*.jpg')))
        if ~allow_download
            error('BSDS500 test images not found in %s; enable auto_download.',folder);
        end
        if ~isfolder(data_dir),mkdir(data_dir);end
        archive=fullfile(data_dir,'BSR_bsds500.tgz');
        url='https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources/BSR_bsds500.tgz';
        if ~isfile(archive)
            fprintf('Downloading BSDS500 archive (large download)...\n');
            try,websave(archive,url);
            catch ME,error('Could not obtain BSDS500. Put BSDS500 JPG files in %s or use Set12. %s',folder,ME.message);end
        end
        untar(archive,data_dir);
    end
    filepath=choose_image(folder,index);
end

function filepath=choose_image(folder,index)
    if ~isfolder(folder),error('Image folder does not exist: %s',folder);end
    extensions={'*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp'};
    names={};
    for jj=1:numel(extensions)
        d=dir(fullfile(folder,extensions{jj}));
        names=[names,{d.name}]; %#ok<AGROW>
    end
    names=sort(names);
    if isempty(names),error('No supported image files in: %s',folder);end
    if index<1 || index>numel(names) || index~=floor(index)
        error('image_index must be between 1 and %d in folder %s.',numel(names),folder);
    end
    filepath=fullfile(folder,names{index});
end

function out=prepare_square_image(raw,N,mode)
    [h,w]=size(raw);
    switch lower(mode)
        case 'preserve_crop'
            % Isotropic scaling, then center crop; no stretch or toolbox.
            scale=max(N/h,N/w);
            new_h=max(N,ceil(h*scale));new_w=max(N,ceil(w*scale));
            enlarged=resize_bilinear(raw,new_h,new_w);
            ri=floor((new_h-N)/2)+1;
            ci=floor((new_w-N)/2)+1;
            out=enlarged(ri:ri+N-1,ci:ci+N-1);
        case 'stretch'
            out=resize_bilinear(raw,N,N);
        otherwise
            error('Unknown cfg.resize_mode: %s',mode);
    end
    out=min(max(out,0),1);
end

function out=resize_bilinear(u,nh,nw)
    [h,w]=size(u);
    % interp2 is core MATLAB. Pixel-center coordinates.
    xq=linspace(1,w,nw);yq=linspace(1,h,nh);
    [X,Y]=meshgrid(xq,yq);
    out=interp2(u,X,Y,'linear');
end

function y=haar2_orth(u,levels,inverse)
    % Orthonormal multilevel Haar transform; forward or inverse.
    % Input/output are N x N square matrices.
    [N,M]=size(u);
    assert(N==M && mod(N,2^levels)==0,'Invalid Haar image size/levels.');
    y=u; rt2=sqrt(2);
    if ~inverse
        for ell=1:levels
            sz=N/2^(ell-1);
            block=y(1:sz,1:sz);
            low=(block(:,1:2:end)+block(:,2:2:end))/rt2;
            high=(block(:,1:2:end)-block(:,2:2:end))/rt2;
            block=[low,high];
            low=(block(1:2:end,:)+block(2:2:end,:))/rt2;
            high=(block(1:2:end,:)-block(2:2:end,:))/rt2;
            y(1:sz,1:sz)=[low;high];
        end
    else
        for ell=levels:-1:1
            sz=N/2^(ell-1);hs=sz/2;
            block=y(1:sz,1:sz);
            reconstructed=zeros(sz,sz);
            reconstructed(1:2:end,:)=(block(1:hs,:)+block(hs+1:end,:))/rt2;
            reconstructed(2:2:end,:)=(block(1:hs,:)-block(hs+1:end,:))/rt2;
            block=reconstructed;
            reconstructed(:,1:2:end)=(block(:,1:hs)+block(:,hs+1:end))/rt2;
            reconstructed(:,2:2:end)=(block(:,1:hs)-block(:,hs+1:end))/rt2;
            y(1:sz,1:sz)=reconstructed;
        end
    end
end

function plot_restoration_three(infos,labels,ref,outdir,save_figures)
    cc=lines(numel(infos));
    linespec={'-','--','-', '--','-','-','-.','-.','--',':'};
    mark={'o','s','d','>','x','p','^','p','h','v'};
    for ptype=1:3
        f=figure('Color','w','Position',[150 120 850 540]);hold on;
        for j=1:numel(infos)
            v=infos{j};
            if ptype==1
                y=v.f_hist-ref;x=(1:numel(y))';
                % Logarithmic y-axis cannot display zero or negative gaps.
                y(~isfinite(y) | y<=0)=NaN;
            elseif ptype==2
                y=v.res_hist;x=(1:numel(y))';
                y(~isfinite(y) | y<=eps(max(1,abs(ref))))=NaN;
            else
                y=v.f_hist-ref;x=v.t_hist;
                y(~isfinite(y) | y<=eps(max(1,abs(ref))))=NaN;
            end
            semilogy(x,y,'Color',cc(j,:),'LineStyle',linespec{j}, ...
                'Marker',mark{j},'MarkerSize',5,'LineWidth',2.0, ...
                'MarkerIndices',1:max(1,ceil(numel(y)/20)):numel(y), ...
                'DisplayName',labels{j});
        end
        grid on;
        switch ptype
            case 1
                xlabel('Iteration Number','Interpreter','latex');
                ylabel('$\phi(x_k)-\phi_{\mathrm{ref}}$','Interpreter','latex');
                ttl='Objective vs Iteration';fname='Objective_vs_Iteration';
            case 2
                xlabel('Iteration Number','Interpreter','latex');
                ylabel('$|\phi(x_{k+1})-\phi(x_k)|/(1+|\phi(x_k)|)$','Interpreter','latex');
                ttl='Residual vs Iteration';fname='Residual_vs_Iteration';
            case 3
                xlabel('Time (s)','Interpreter','latex');
                ylabel('$\phi(x_k)-\phi_{\mathrm{ref}}$','Interpreter','latex');
                ttl='Objective vs Time';fname='Objective_vs_Time';
        end
        title(ttl,'FontSize',18);
        set(gca,'FontSize',16,'LineWidth',1);
        legend('show','Location','northeast','FontSize',11,'Box','off');
        if save_figures
            exportgraphics(f,fullfile(outdir,[fname,'.png']),'Resolution',200);
            exportgraphics(f,fullfile(outdir,[fname,'.pdf']));
        end
    end
end
