function test_apply_gamma_matrices
% TEST_APPLY_GAMMA_MATRICES
% A/B equivalence between the two Gamma paths used by
% fwd_solve_1st_order_mdeit's magnetic-field section:
%   - compute_gamma_matrices + Gamma*u   (forms Gamma explicitly)
%   - apply_gamma_matrices(img, u)       (matrix-free, never forms Gamma)
%
% Covered, for both mdeit1 and mdeit3:
%   1. img.fwd_model.mdeit.matrix_free = false/true select the two paths and
%      agree to ~1e-12 (not bitwise: summation order differs).
%   2. The small model used here has n_stim > 1 (multi-stimulation), which is
%      the case most likely to expose a bug in the [py,pz] stacking used to
%      halve R traffic in apply_gamma_matrices.
%
% Run from anywhere:   test_apply_gamma_matrices
% Exits with an error (non-zero) if any check fails, so it is CI-friendly.

fullpath      = mfilename('fullpath');
tests_folder  = fileparts(fullpath);
workspace_dir = fileparts(tests_folder);
addpath(genpath(fullfile(workspace_dir,'functions')));
setup_eidors(workspace_dir);

tol = 1e-12;

fprintf('== test_apply_gamma_matrices ==\n');

n_fail = 0;
n_fail = n_fail + run_case('mdeit3', tol);
n_fail = n_fail + run_case('mdeit1', tol);

if n_fail > 0
    error('test_apply_gamma_matrices: %d check(s) FAILED', n_fail);
end
fprintf('ALL PASSED\n');

end


function n_fail = run_case(recon_mode, tol)
n_fail = 0;
fprintf('\n-- recon_mode = %s --\n', recon_mode);

img = build_small_mdeit_model(recon_mode);
fprintf('  n_stim = %d, n_sensors = %d\n', ...
    numel(img.fwd_model.stimulation), numel(img.fwd_model.sensors));

img.fwd_model.mdeit.matrix_free = false;
d_ref = fwd_solve_1st_order_mdeit(img);

img.fwd_model.mdeit.matrix_free = true;
d_mf  = fwd_solve_1st_order_mdeit(img);

rel = relerr(d_mf.meas, d_ref.meas);
n_fail = n_fail + report(rel < tol, 'matrix-free meas == forming-Gamma meas', rel);

end


%% ------------------------------------------------------------------------
function img = build_small_mdeit_model(recon_mode)
% A small cylinder mDEIT model with multiple stimulation patterns (n_stim>1
% is the case most likely to expose a bug in the matrix-free RHS stacking).
% Mirrors build_small_mdeit_model in test_fwd_solve_sys_mat_mdeit.m.

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
fprintf('  [%s] %-40s  err=%.2e\n', pf(ok), name, err);
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
