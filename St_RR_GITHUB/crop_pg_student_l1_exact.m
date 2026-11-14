function [x,info,acc] = crop_pg_student_l1_exact(problem,x0,opts)
% Wrapper for the CROP-Exact method in the E2006 Student-t experiment.
% All underlying implementation is in student_solver.m.
    opts.method = 'CROP-Exact';
    [x,info] = student_solver_optimized(problem,x0,opts);
    acc = info.n_accepted;
end
