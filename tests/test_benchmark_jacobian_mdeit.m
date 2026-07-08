function test_benchmark_jacobian_mdeit
% TEST_BENCHMARK_JACOBIAN_MDEIT
% Speed comparison of the two ways to apply the mDEIT Jacobian:
%
%   (A) dense assembly      J   = calc_jacobian_mdeit(img);      then  J*v
%   (B) matrix-free operator Jop = calc_jacobian_operator_mdeit(img); Jop.mtimes(v)
%
% For a Jacobian-vector product (and its transpose) it reports, per model:
%   - one-off BUILD time of J vs Jop
%   - per-matvec time  J*v      vs  Jop.mtimes(v)
%   - per-matvec time  J'*w     vs  Jop.mtimes_transp(w)
%   - the dense J memory footprint
%   - the crossover N: how many matvecs before the cheaper *build* of the
%     operator is outweighed by its more expensive matvecs (i.e. when does
%     dense become the faster total for N products).
%
% It also asserts the two approaches agree (so this is a real test, not
% only a benchmark): it errors out if any J*v / J'*w disagree.
%
% Run from anywhere:   test_benchmark_jacobian_mdeit

%% Prepare workspace ------------------------------------------------------
fullpath      = mfilename('fullpath');
tests_folder  = fileparts(fullpath);
workspace_dir = fileparts(tests_folder);
addpath(genpath(fullfile(workspace_dir,'functions')));
setup_eidors(workspace_dir);

rng(0);
tol = 1e-8;                  % equivalence tolerance (reordered FP sums)

fprintf('\n================ BENCHMARK: mDEIT Jacobian-vector product ================\n');
fprintf('MATLAB %s   |   maxNumCompThreads = %d\n', version, maxNumCompThreads);

% A couple of small problem sizes so the trend with mesh size is visible.
cases = { ...
    struct('mode','mdeit3','maxsz',0.40,'label','small' ), ...
    struct('mode','mdeit3','maxsz',0.25,'label','medium'), ...
    struct('mode','mdeit1','maxsz',0.40,'label','small' ), ...
    struct('mode','mdeit1','maxsz',0.25,'label','medium') };

n_fail = 0;
for i = 1:numel(cases)
    n_fail = n_fail + run_case(cases{i}, tol);
end

fprintf('\n==========================================================================\n');
if n_fail > 0
    error('test_benchmark_jacobian_mdeit: %d equivalence check(s) FAILED', n_fail);
end
fprintf('All equivalence checks PASSED.\n');
end


%% ------------------------------------------------------------------------
function n_fail = run_case(c, tol)
n_fail = 0;

img    = build_small_mdeit_model(c.mode, c.maxsz);
n_elem = size(img.fwd_model.elems,1);
n_node = size(img.fwd_model.nodes,1);

fprintf('\n-- %s / %s : %d elems, %d nodes -----------------------------------\n', ...
    c.mode, c.label, n_elem, n_node);

% Warm up (mesh gen, eidors caches, JIT) so build timings are steady-state.
J0   = calc_jacobian_mdeit(img);           %#ok<NASGU>
Jop0 = calc_jacobian_operator_mdeit(img);  %#ok<NASGU>

% ---- one-off BUILD times (min of a few reps) ----
t_build_dense = time_min(@() calc_jacobian_mdeit(img),          3);
t_build_op    = time_min(@() calc_jacobian_operator_mdeit(img), 3);

% Build the two objects we will hammer with matvecs.
J   = calc_jacobian_mdeit(img);
Jop = calc_jacobian_operator_mdeit(img);

n_meas = size(J,1);
n_col  = size(J,2);
v = randn(n_col,1);
w = randn(n_meas,1);

% ---- equivalence (this is the "test" part) ----
e_fwd = relerr(Jop.mtimes(v),        J*v);
e_trn = relerr(Jop.mtimes_transp(w), J.'*w);
ok = (e_fwd < tol) && (e_trn < tol);
n_fail = n_fail + ~ok;
fprintf('   equivalence: J*v relerr=%.2e   J''*w relerr=%.2e   [%s]\n', ...
    e_fwd, e_trn, pf(ok));

% ---- per-matvec times (timeit: robust, auto-warmup/averaged) ----
t_mv_dense  = timeit(@() J*v);
t_mv_op     = timeit(@() Jop.mtimes(v));
t_mvt_dense = timeit(@() J.'*w);
t_mvt_op    = timeit(@() Jop.mtimes_transp(w));

memJ = numel(J)*8/1e6;   % dense J footprint (double), MB

fprintf('   build   :  dense %8.2f ms   operator %8.2f ms   (op/dense = %.2fx)\n', ...
    1e3*t_build_dense, 1e3*t_build_op, t_build_op/t_build_dense);
fprintf('   J *v    :  dense %8.1f us   operator %8.1f us   (op/dense = %.1fx)\n', ...
    1e6*t_mv_dense, 1e6*t_mv_op, t_mv_op/t_mv_dense);
fprintf('   J''*w    :  dense %8.1f us   operator %8.1f us   (op/dense = %.1fx)\n', ...
    1e6*t_mvt_dense, 1e6*t_mvt_op, t_mvt_op/t_mvt_dense);
fprintf('   dense J memory: %.2f MB  (%d x %d doubles)\n', memJ, n_meas, n_col);

% ---- crossover: total time for N forward matvecs incl. build ----
%   dense_total(N)    = t_build_dense + N*t_mv_dense
%   operator_total(N) = t_build_op    + N*t_mv_op
report_crossover(t_build_dense, t_mv_dense, t_build_op, t_mv_op);
end


%% ------------------------------------------------------------------------
function report_crossover(bD, mD, bO, mO)
% Total time for N forward matvecs, incl. build:
%   dense_total(N)    = bD + N*mD
%   operator_total(N) = bO + N*mO
% Their difference diff(N) = (bO-bD) + N*(mO-mD); operator is cheaper while
% diff(N) < 0. Crossover at N0 = (bD-bO)/(mO-mD).
denom = mO - mD;
if abs(denom) < eps*max([mO, mD, 1])
    winner = ternary(bO < bD, 'operator', 'dense');
    fprintf('   crossover: matvec cost ~equal; %s wins (cheaper build).\n', winner);
    return
end
N0 = (bD - bO) / denom;
if N0 <= 0
    % No positive crossover: one method is cheaper in BOTH build and matvec.
    winner = ternary(bO < bD, 'operator', 'dense');
    fprintf('   crossover: %s wins at every N (cheaper build AND cheaper matvec).\n', winner);
    return
end
% Positive crossover: the method with the cheaper build wins below N0.
lo = ternary(bO < bD, 'operator', 'dense');
hi = ternary(bO < bD, 'dense', 'operator');
fprintf('   crossover: %s faster up to ~%.0f matvecs; beyond that %s wins.\n', lo, N0, hi);
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end


%% ------------------------------------------------------------------------
function t = time_min(fn, reps)
% Minimum wall time over reps (best-case, least noisy for one-off builds).
t = inf;
for k = 1:reps
    tic; fn(); t = min(t, toc);
end
end


%% ------------------------------------------------------------------------
function img = build_small_mdeit_model(recon_mode, maxsz)
% A small cylinder mDEIT model, mirroring test.m but coarse for speed.

height = 3; radius = 1.0;
num_electrodes_ring = 4; num_rings = 2;
ring_vert_pos = height/(num_rings+1):height/(num_rings+1):height-height/(num_rings+1);
electrode_radius = 0.3;

elec_pos   = [num_electrodes_ring, ring_vert_pos];
elec_shape = [electrode_radius, electrode_radius, maxsz];

fmdl = ng_mk_cyl_models([height,radius,maxsz], elec_pos, elec_shape);

stim = mk_stim_patterns(num_electrodes_ring, num_rings, [0,1], [0,1], ...
                        {'meas_current'}, 1.0);
fmdl.stimulation = stim;
for i = 1:num_electrodes_ring*num_rings
    fmdl.electrode(i).z_contact = 1.0;
end

% Magnetometers
num_sensors_ring = num_electrodes_ring;
num_sensors = num_sensors_ring*num_rings;
thetam = 0:2*pi/num_sensors_ring:2*pi-2*pi/num_sensors_ring;
sensor_radius = 1.5*radius;
sensor_positions = zeros(num_sensors,3);
for i = 1:num_rings
    for m = 1:num_sensors_ring
        sensor_positions((i-1)*num_sensors_ring+m,:) = ...
            [sensor_radius*cos(thetam(m)), sensor_radius*sin(thetam(m)), ring_vert_pos(i)];
    end
end

switch recon_mode
    case 'mdeit3'
        sensor_axes = struct('axis1',[1,0,0],'axis2',[0,1,0],'axis3',[0,0,1]);
    case 'mdeit1'
        sensor_axes = struct('axis',[0,0,1]);
    otherwise
        error('unknown recon_mode %s', recon_mode);
end
fmdl = assign_magnetometers(fmdl, sensor_positions, sensor_axes);

mu0  = 1;
fmdl = compute_geometry_matrices(fmdl, mu0);

img = mk_image(fmdl, 1.0);
end


%% ------------------------------------------------------------------------
function e = relerr(a, b)
d = norm(a(:) - b(:));
n = norm(b(:));
if n == 0, n = 1; end
e = d / n;
end

function s = pf(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
