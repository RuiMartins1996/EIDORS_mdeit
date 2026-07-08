function test_inv_solve_diff_GN_one_step_mdeit
% TEST_INV_SOLVE_DIFF_GN_ONE_STEP_MDEIT
% Test suite for the differential one-step Gauss-Newton mDEIT solver.
%
% inv_solve_diff_GN_one_step_mdeit solves the regularized normal equations
%     (J'*J + hp^2*RtR) x = J'*dm
% for a conductivity-change image from magnetic-field difference data. It has
% two independent solve paths that must agree:
%   - dense  left_divide  (default)
%   - matrix-free PCG     (inv_model.inv_solve_core.do_pcg = true)
% and it expands vector data to a matrix / supports multi-frame (MxT) data,
% producing one image column per time step.
%
% Cases (run for both mdeit3 and mdeit1 sensor modes):
%   1. Structural output      - shape/name/fwd_model of returned image
%   2. PCG vs left_divide      - the two paths solve the same system (primary)
%   3. Multi-frame data        - MxT data -> T image columns, per-column match
%   4. Vector->matrix expansion- vector data1 + MxT data2 -> n_elem x T
%   5. Anomaly localization    - peak of reconstruction near true anomaly
%
% Run from anywhere:   test_inv_solve_diff_GN_one_step_mdeit
% Exits with an error (non-zero) if any check fails, so it is CI-friendly.

%% Prepare workspace ------------------------------------------------------
fullpath      = mfilename('fullpath');
tests_folder  = fileparts(fullpath);
workspace_dir = fileparts(tests_folder);
addpath(genpath(fullfile(workspace_dir,'functions')));
setup_eidors(workspace_dir);

rng(0);                      % reproducible

fprintf('== test_inv_solve_diff_GN_one_step_mdeit ==\n');

n_fail = 0;
n_fail = n_fail + run_all_cases('mdeit3');
n_fail = n_fail + run_all_cases('mdeit1');

if n_fail > 0
    error('test_inv_solve_diff_GN_one_step_mdeit: %d check(s) FAILED', n_fail);
end
fprintf('\nALL PASSED\n');

end


%% ------------------------------------------------------------------------
function n_fail = run_all_cases(recon_mode)
n_fail = 0;
fprintf('\n-- recon_mode = %s --\n', recon_mode);

bkgnd = 1.0;
img    = build_small_mdeit_model(recon_mode);
fmdl   = img.fwd_model;
imdl   = build_inv_model(fmdl, bkgnd);

% A single, well-separated spherical anomaly used by most cases.
anomaly = struct('pos',[0.5, 0, 1.5], 'radius',0.35, 'delta',1.0);
[data1, data2] = make_diff_data(fmdl, bkgnd, anomaly);

%% Case 1: structural output ---------------------------------------------
img_out = inv_solve_diff_GN_one_step_mdeit(imdl, data1, data2);
n_elem  = size(fmdl.elems,1);

ok = isfield(img_out,'elem_data') ...
     && isequal(size(img_out.elem_data), [n_elem, 1]) ...
     && isfield(img_out,'name') && ~isempty(img_out.name) ...
     && isfield(img_out,'fwd_model') ...
     && isequal(size(img_out.fwd_model.elems), size(fmdl.elems));
n_fail = n_fail + ~ok;
fprintf('  [%s] structural output (elem_data %dx1, name, fwd_model)\n', ...
    pf(ok), n_elem);

%% Case 2: PCG vs left_divide equivalence (primary) ----------------------
imdl_ld  = imdl;                        % dense left_divide (default)
imdl_pcg = imdl;
imdl_pcg.inv_solve_core.do_pcg = true;
imdl_pcg.inv_solve_core.tol    = 1e-12; % tight -> match dense solve
imdl_pcg.inv_solve_core.maxit  = 5000;

sol_ld  = get_sol(inv_solve_diff_GN_one_step_mdeit(imdl_ld,  data1, data2));
sol_pcg = get_sol(inv_solve_diff_GN_one_step_mdeit(imdl_pcg, data1, data2));

e_equiv = relerr(sol_pcg, sol_ld);
ok = e_equiv < 1e-6;
n_fail = n_fail + ~ok;
fprintf('  [%s] PCG vs left_divide  relerr=%.2e\n', pf(ok), e_equiv);

%% Case 3: multi-frame data (MxT -> T columns) ---------------------------
% Column-scaled frames: solver is linear in dm, so column k must equal the
% single-frame solve of column k. Validates the matrix (multi-rhs) path.
scales = [1.0, 2.0, -0.5];
D1 = repmat(data1(:), 1, numel(scales));
D2 = data1(:) + (data2(:) - data1(:)) .* scales;   % dm_k = scale_k * dm_1

img_multi = inv_solve_diff_GN_one_step_mdeit(imdl, D1, D2);
ok_shape  = isequal(size(img_multi.elem_data), [n_elem, numel(scales)]);

e_col = 0;
for k = 1:numel(scales)
    sol_k = get_sol(inv_solve_diff_GN_one_step_mdeit(imdl, D1(:,k), D2(:,k)));
    e_col = max(e_col, relerr(img_multi.elem_data(:,k), sol_k));
end
ok = ok_shape && (e_col < 1e-10);
n_fail = n_fail + ~ok;
fprintf('  [%s] multi-frame (T=%d) shape=%d  max col relerr=%.2e\n', ...
    pf(ok), numel(scales), ok_shape, e_col);

%% Case 4: vector -> matrix expansion ------------------------------------
% data1 a single vector, data2 an MxT matrix: data1 must expand to MxT.
img_exp = inv_solve_diff_GN_one_step_mdeit(imdl, data1(:), D2);
ok = isequal(size(img_exp.elem_data), [n_elem, numel(scales)]);
% and the expanded result must match passing the explicit expanded data1
e_exp = relerr(img_exp.elem_data, img_multi.elem_data);
ok = ok && (e_exp < 1e-10);
n_fail = n_fail + ~ok;
fprintf('  [%s] vector->matrix expansion  relerr=%.2e\n', pf(ok), e_exp);

%% Case 5: anomaly localization ------------------------------------------
% Only the full 3-axis mode has enough field sensitivity to reliably
% localize an off-axis anomaly with a one-step linear reconstruction on a
% coarse mesh. The single-component (z-only) mdeit1 mode is degenerate for
% this, so we assert localization for mdeit3 only.
if strcmp(recon_mode, 'mdeit3')
    sol = get_sol(img_out);             % from case 1 (single anomaly)
    [~, idx_pk] = max(abs(sol));
    centroids   = elem_centroids(fmdl);
    pk_pos      = centroids(idx_pk, :);
    dist        = norm(pk_pos - anomaly.pos);

    loc_tol = 3 * anomaly.radius;       % generous: coarse mesh, one step
    ok_loc  = dist < loc_tol;
    ok_sign = sol(idx_pk) > 0;          % conductivity increase (delta>0)
    ok = ok_loc && ok_sign;
    n_fail = n_fail + ~ok;
    fprintf('  [%s] localization  peak dist=%.2f (tol %.2f)  sign ok=%d\n', ...
        pf(ok), dist, loc_tol, ok_sign);
else
    fprintf('  [SKIP] localization (not asserted for single-axis %s)\n', ...
        recon_mode);
end

end


%% ------------------------------------------------------------------------
function imdl = build_inv_model(fmdl, bkgnd)
% Inverse model built the way test.m does (difference reconstruction).
imdl.type        = 'inv_model';
imdl.name        = 'mdeit diff GN one-step test model';
imdl.reconst_type= 'difference';
imdl.fwd_model   = fmdl;
imdl.solve       = @inv_solve_diff_GN_one_step_mdeit;

img_bkgnd = mk_image(fmdl, bkgnd);
imdl.jacobian_bkgnd.elem_data = img_bkgnd.elem_data;

imdl.RtR_prior          = @prior_noser;
imdl.hyperparameter.value = 0.005;

imdl.inv_solve_core.print_diagnostics = false;
end


%% ------------------------------------------------------------------------
function [data1, data2] = make_diff_data(fmdl, bkgnd, anomaly)
% Homogeneous (data1) and anomaly (data2) mDEIT difference data.
img1 = mk_image(fmdl, bkgnd);

p = anomaly.pos; r = anomaly.radius;
select_fcn = @(x,y,z) (x-p(1)).^2 + (y-p(2)).^2 + (z-p(3)).^2 < r^2;
memb_frac  = elem_select(img1.fwd_model, select_fcn);
img2 = mk_image(img1, bkgnd*(1 + anomaly.delta*memb_frac));

data1 = fwd_solve_1st_order_mdeit(img1);
data2 = fwd_solve_1st_order_mdeit(img2);
data1 = data1.meas;
data2 = data2.meas;
end


%% ------------------------------------------------------------------------
function img = build_small_mdeit_model(recon_mode)
% A small cylinder mDEIT model, mirroring test.m but coarse for speed.
% (Same builder as test_calc_jacobian_operator_mdeit.)

height = 3; radius = 1.0; maxsz = 0.4;
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
function c = elem_centroids(fmdl)
% Element centroids as the mean of their node coordinates.
n_elem = size(fmdl.elems,1);
c = zeros(n_elem, size(fmdl.nodes,2));
for d = 1:size(fmdl.nodes,2)
    coord = fmdl.nodes(:,d);
    c(:,d) = mean(coord(fmdl.elems), 2);
end
end


%% ------------------------------------------------------------------------
function s = get_sol(img)
% Extract solution vector/matrix from a returned image.
s = img.elem_data;
end

function e = relerr(a, b)
d = norm(a(:) - b(:));
n = norm(b(:));
if n == 0, n = 1; end
e = d / n;
end

function s = pf(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
