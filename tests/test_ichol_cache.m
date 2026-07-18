function test_ichol_cache
% TEST_ICHOL_CACHE
% Phase 4 of CACHE_PLAN.md / ICHOL_CACHE_PLAN.md: build_ichol (the pcg
% preconditioner in fwd_solve_1st_order_mdeit) is cached via eidors_cache,
% keyed on E alone. A second fwd_solve_1st_order_mdeit call with the same E
% (same mesh/conductivity) must not rebuild it, and both calls must produce
% bitwise-identical data.meas/data.volt.
%
% Run from anywhere:   test_ichol_cache
% Exits with an error (non-zero) if any check fails, so it is CI-friendly.

%% Prepare workspace ------------------------------------------------------
fullpath      = mfilename('fullpath');
tests_folder  = fileparts(fullpath);
workspace_dir = fileparts(tests_folder);
addpath(genpath(fullfile(workspace_dir,'functions')));
setup_eidors(workspace_dir);

fprintf('== test_ichol_cache ==\n');

n_fail = 0;
n_fail = n_fail + run_case('mdeit1');
n_fail = n_fail + run_case('mdeit3');

if n_fail > 0
    error('test_ichol_cache: %d check(s) FAILED', n_fail);
end
fprintf('ALL PASSED\n');

end


%% ------------------------------------------------------------------------
function n_fail = run_case(recon_mode)
n_fail = 0;
fprintf('\n-- recon_mode = %s --\n', recon_mode);

img = build_small_mdeit_model(recon_mode);
img.fwd_model.solve_mdeit.method = 'pcg';
img.fwd_solve.get_all_meas = 1;   % also compare data.volt

eidors_cache('clear_all');
eidors_cache('debug_on', 'build_ichol');

out1 = evalc('d1 = fwd_solve_1st_order_mdeit(img);');   % cold: must build ichol
out2 = evalc('d2 = fwd_solve_1st_order_mdeit(img);');   % warm: must reuse it

eidors_cache('debug_off', 'build_ichol');

built1 = ~isempty(strfind(out1, 'Building iChol preconditioner'));
built2 = ~isempty(strfind(out2, 'Building iChol preconditioner'));

n_fail = n_fail + report(built1,  'cold call builds ichol',        ~built1);
n_fail = n_fail + report(~built2, 'warm call reuses cached ichol', built2);

ok_meas = isequal(d1.meas, d2.meas);
n_fail = n_fail + report(ok_meas, 'data.meas identical (bitwise)', ~ok_meas);

ok_volt = isequal(d1.volt, d2.volt);
n_fail = n_fail + report(ok_volt, 'data.volt identical (bitwise)', ~ok_volt);

end


%% ------------------------------------------------------------------------
function img = build_small_mdeit_model(recon_mode)
% A small cylinder mDEIT model, mirroring test_fwd_solve_sys_mat_mdeit.m's
% builder of the same name, but coarse for speed.

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

function s = pf(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
