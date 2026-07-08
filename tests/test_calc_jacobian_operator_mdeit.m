function test_calc_jacobian_operator_mdeit
% TEST_CALC_JACOBIAN_OPERATOR_MDEIT
% Equivalence test for the matrix-free mDEIT Jacobian operator.
%
% Verifies that calc_jacobian_operator_mdeit (which never assembles J and
% streams the element contraction in blocks) reproduces the dense Jacobian
% from calc_jacobian_mdeit, for both the 3-axis and 1-axis reconstruction
% modes:
%
%   Jop.mtimes(v)        == J  * v      (for random v)
%   Jop.mtimes_transp(w) == J' * w      (for random w)
%
% A deliberately tiny chunk_size is also exercised so the block loop runs
% many iterations, catching any block-boundary / accumulation bug.
%
% Run from anywhere:   test_calc_jacobian_operator_mdeit
% Exits with an error (non-zero) if any check fails, so it is CI-friendly.

%% Prepare workspace ------------------------------------------------------
fullpath      = mfilename('fullpath');
tests_folder  = fileparts(fullpath);
workspace_dir = fileparts(tests_folder);
addpath(genpath(fullfile(workspace_dir,'functions')));
setup_eidors(workspace_dir);

rng(0);                      % reproducible random probe vectors
tol = 1e-8;                  % relative tolerance (same math, reordered sums)

fprintf('== test_calc_jacobian_operator_mdeit ==\n');

n_fail = 0;
n_fail = n_fail + run_case('mdeit3', tol);
n_fail = n_fail + run_case('mdeit1', tol);

if n_fail > 0
    error('test_calc_jacobian_operator_mdeit: %d check(s) FAILED', n_fail);
end
fprintf('ALL PASSED\n');

end


%% ------------------------------------------------------------------------
function n_fail = run_case(recon_mode, tol)
n_fail = 0;
fprintf('\n-- recon_mode = %s --\n', recon_mode);

img = build_small_mdeit_model(recon_mode);

% Dense reference Jacobian
J = calc_jacobian_mdeit(img);

% Test with the automatic chunk size AND a tiny forced one (many blocks)
n_elem = size(img.fwd_model.elems,1);
chunk_sizes = {[], 7, n_elem};   % [] => automatic ; 7 => many small blocks

for cs = chunk_sizes
    img_c = img;
    label = 'auto';
    if ~isempty(cs{1})
        img_c.fwd_model.jacobian_operator.chunk_size = cs{1};
        label = sprintf('chunk=%d', cs{1});
    end

    Jop = calc_jacobian_operator_mdeit(img_c);

    % size
    if ~isequal(Jop.size, size(J))
        fprintf('  [FAIL] %-10s size mismatch: op=[%d %d] dense=[%d %d]\n', ...
            label, Jop.size(1), Jop.size(2), size(J,1), size(J,2));
        n_fail = n_fail + 1;
        continue
    end

    n_meas = size(J,1);
    n_col  = size(J,2);

    % forward:  J*v
    v  = randn(n_col,1);
    e_fwd = relerr(Jop.mtimes(v), J*v);

    % transpose:  J'*w
    w  = randn(n_meas,1);
    e_trn = relerr(Jop.mtimes_transp(w), J.'*w);

    ok_fwd = e_fwd < tol;
    ok_trn = e_trn < tol;
    n_fail = n_fail + ~ok_fwd + ~ok_trn;

    fprintf('  [%s] %-10s  J*v relerr=%.2e   J''*w relerr=%.2e\n', ...
        pf(ok_fwd && ok_trn), label, e_fwd, e_trn);
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
function e = relerr(a, b)
d = norm(a(:) - b(:));
n = norm(b(:));
if n == 0, n = 1; end
e = d / n;
end

function s = pf(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
