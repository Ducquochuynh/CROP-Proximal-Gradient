function plot_original_three(infos,labels,phi_ref,output_dir,save_figures)
    colors=lines(numel(infos));
    styles={'-','--','-','--','-','-','-.','-.','--'};
    markers={'o','s','d','>','x','p','^','p','h'};
    fig_pos=[200 200 800 500];
    floor_gap=eps(max(1,abs(phi_ref)));  % only for log visualization
    for fig_id=1:3
        switch fig_id
            case 1
                figtitle='Objective vs Iteration';
                x_label='Iteration Number';
                y_label='$\phi(x_k)-\phi_{\mathrm{ref}}$';
                outname='E2006_Objective_vs_Iteration';
            case 2
                figtitle='Residual vs Iteration';
                x_label='Iteration Number';
                y_label='$|\phi(x_{k+1})-\phi(x_k)|/(1+|\phi(x_k)|)$';
                outname='E2006_Residual_vs_Iteration';
            case 3
                figtitle='Objective vs Time';
                x_label='Time (s)';
                y_label='$\phi(x_k)-\phi_{\mathrm{ref}}$';
                outname='E2006_Objective_vs_Time';
        end
        f=figure('Name',figtitle,'Color','w');
        set(f,'Position',fig_pos);
        hold on;
        for j=1:numel(infos)
            infj=infos{j};
            if fig_id==1
                xp=1:numel(infj.f_hist);
                yp=max(infj.f_hist-phi_ref,floor_gap);
            elseif fig_id==2
                xp=1:numel(infj.res_hist);
                yp=max(infj.res_hist,realmin);
            else
                xp=infj.t_hist;
                yp=max(infj.f_hist-phi_ref,floor_gap);
            end
            semilogy(xp,yp,'Color',colors(j,:),'LineStyle',styles{j}, ...
                'Marker',markers{j},'MarkerSize',5,'LineWidth',2.2, ...
                'DisplayName',labels{j});
        end
        grid on;
        xlabel(x_label,'Interpreter','latex','FontSize',20,'FontWeight','bold');
        ylabel(y_label,'Interpreter','latex','FontSize',20);
        title(figtitle,'FontSize',20,'FontWeight','bold');
        legend('show','Location','northeast','FontSize',15,'Box','off');
        set(gca,'FontSize',20,'LineWidth',1);
        if save_figures
            exportgraphics(f,fullfile(output_dir,[outname '.png']),'Resolution',200);
            exportgraphics(f,fullfile(output_dir,[outname '.pdf']),'ContentType','vector');
        end
    end
end
