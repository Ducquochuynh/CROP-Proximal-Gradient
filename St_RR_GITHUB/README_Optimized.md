# E2006 Student-t robust L1: optimized nine-method package

Run `Run_St_RR.m` .

Set `cfg.csafe=1` to enable the CROP consistency safeguard `norm(ycr-z_k)<=c*norm(x_k-x_{k-1})`, or `Inf` to disable it. 

The original objective safeguard, inexact `Hk*alpha` versus exact true residual, all ten updates, relative objective-change stopping, E2006 sparse loader, and three original plot styles are retained. Buffers are preallocated to avoid repeated allocation of huge sparse-feature-dimension columns. Solver histories use one index per proximal update including the first PG step.

`info.k` counts updates, `info.elapsed_time` and `info.t_hist` are wall-clock seconds, `info.cpu_time` is process CPU seconds. Summary reports both. Dataset downloading, parsing, norm calculation, optional test evaluation, and plots are excluded from solver times. Note that CPU time may exceed wall-clock time with multithreading.

Run `Test_StudentT_Sanity.m` for a local, no-download smoke test of the modular solver. MATLAB benchmarks have NOT been run here. The Student-t objective is nonconvex; `phi_ref` is only the best observed objective and the objective-change stopping condition does not prove stationarity.
