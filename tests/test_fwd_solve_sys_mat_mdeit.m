function test_fwd_solve_sys_mat_mdeit
% TEST_FWD_SOLVE_SYS_MAT_MDEIT
% Tests the "solve + system matrix in one pass" refactor and the optional
% skip_mag_field switch of the mDEIT forward solver.
%
% Covered:
%   1. fwd_solve_sys_mat_mdeit returns s_mat_E identical to lhs_eit_full(img),
%      i.e. the system matrix is assembled once and handed back.
%   2. data.volt from the combined call matches the standard EIT node solve.
%   3. skip_mag_field = true skips compute_gamma_matrices: data.meas comes
%      back empty, while data.volt and s_mat_E are byte-for-byte the same as
%      the non-skipping call.
%   4. With skip_mag_field = false the returned data.meas equals the magnetic
%      field computed independently from Gamma * volt.
%   5. End-to-end: calc_jacobian_operator_mdeit (whose 3-axis setup now runs
%      the forward solve with skip_mag_field = true) still reproduces the
%      dense Jacobian from calc_jacobian_mdeit.
%
% Run from anywhere:   test_fwd_solve_sys_mat_mdeit
% Exits with an error (non-zero) if any check fails, so it is CI-friendly.

%% Prepare workspace ------------------------------------------------------
fullpath      = mfilename('fullpath');
tests_folder  = fileparts(fullpath);
workspace_dir = fileparts(tests_folder);
addpath(genpath(fullfile(workspace_dir,'functions')));
setup_eidors(workspace_dir);

rng(0);                      % reproducible random probe vectors
tol_mat = 1e-12;             % same matrix, same call -> essentially identical
tol     = 1e-8;              % reordered floating-point sums

fprintf('== test_fwd_solve_sys_mat_mdeit ==\n');

n_fail = 0;
n_fail = n_fail + run_case('mdeit3', tol_mat, tol);
n_fail = n_fail + run_case('mdeit1', tol_mat, tol);

if n_fail > 0
    error('test_fwd_solve_sys_mat_mdeit: %d check(s) FAILED', n_fail);
end
fprintf('ALL PASSED\n');

end


%% ------------------------------------------------------------------------
function n_fail = run_case(recon_mode, tol_mat, tol)
n_fail = 0;
fprintf('\n-- recon_mode = %s --\n', recon_mode);

img = build_small_mdeit_model(recon_mode);
img.fwd_solve.get_all_meas = 1;         % ask for data.volt as well

%% 1. s_mat_E == lhs_eit_full(img) ---------------------------------------
[data_full, s_mat_E] = fwd_solve_sys_mat_mdeit(img);        % default: no skip
A_ref = lhs_eit_full(img);
e_mat = relerr(s_mat_E, A_ref);
n_fail = n_fail + report(e_mat < tol_mat, 's_mat_E == lhs_eit_full', e_mat);

%% 2. data.volt == standard EIT node solve -------------------------------
u_ref = fwd_solve(img);                 % dispatcher -> standard 1st-order solve
e_volt = relerr(data_full.volt, u_ref.volt);
n_fail = n_fail + report(e_volt < tol, 'data.volt == standard EIT solve', e_volt);

%% 3. skip_mag_field: meas empty, volt & s_mat_E unchanged ---------------
[data_skip, s_mat_E_skip] = fwd_solve_sys_mat_mdeit(img, true);

ok_empty = isempty(data_skip.meas);
n_fail = n_fail + report(ok_empty, 'skip -> data.meas is empty', ~ok_empty);

ok_full_meas = ~isempty(data_full.meas);
n_fail = n_fail + report(ok_full_meas, 'no-skip -> data.meas is populated', ~ok_full_meas);

e_volt_skip = relerr(data_skip.volt, data_full.volt);
n_fail = n_fail + report(e_volt_skip < tol_mat, 'skip -> data.volt unchanged', e_volt_skip);

e_mat_skip = relerr(s_mat_E_skip, s_mat_E);
n_fail = n_fail + report(e_mat_skip < tol_mat, 'skip -> s_mat_E unchanged', e_mat_skip);

%% 4. non-skip data.meas == independent Gamma * volt ---------------------
B_ref = magnetic_field(img, data_full.volt, recon_mode);
e_meas = relerr(data_full.meas, B_ref);
n_fail = n_fail + report(e_meas < tol, 'data.meas == Gamma*volt', e_meas);

%% 5. Jacobian operator (skips the field internally) == dense J ----------
J   = calc_jacobian_mdeit(img);
Jop = calc_jacobian_operator_mdeit(img);

if ~isequal(Jop.size, size(J))
    fprintf('  [FAIL] Jop size mismatch: op=[%d %d] dense=[%d %d]\n', ...
        Jop.size(1), Jop.size(2), size(J,1), size(J,2));
    n_fail = n_fail + 1;
else
    v = randn(size(J,2),1);
    w = randn(size(J,1),1);
    e_fwd = relerr(Jop.mtimes(v),        J*v);
    e_trn = relerr(Jop.mtimes_transp(w), J.'*w);
    n_fail = n_fail + report(e_fwd < tol, 'Jop.mtimes == J*v',        e_fwd);
    n_fail = n_fail + report(e_trn < tol, 'Jop.mtimes_transp == J''*w', e_trn);
end
end


%% ------------------------------------------------------------------------
function B = magnetic_field(img, u, recon_mode)
% Independent reference for data.meas: the magnetic field on the sensors,
% built straight from the Gamma matrices and the FEM node voltages u.
img = compute_gamma_matrices(img);
switch recon_mode
    case 'mdeit1'
        B = img.Gamma*u;
        B = B(:);
    case 'mdeit3'
        Bx = img.Gamma1*u;
        By = img.Gamma2*u;
        Bz = img.Gamma3*u;
        B = [Bx(:); By(:); Bz(:)];
    otherwise
        error('unknown recon_mode %s', recon_mode);
end
end


%% ------------------------------------------------------------------------
function img = build_small_mdeit_model(recon_mode)
% A small cylinder mDEIT model, mirroring test.m but coarse for speed.

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
function n_bad = report(ok, name, err)
% Print a PASS/FAIL line and return 1 when the check failed, 0 otherwise.
fprintf('  [%s] %-34s  err=%.2e\n', pf(ok), name, err);
n_bad = ~ok;
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
