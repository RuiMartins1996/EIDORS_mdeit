function bench_fwd_solve_cache()
% BENCH_FWD_SOLVE_CACHE Benchmark: repeated fwd_solve_1st_order_mdeit on the
% SAME img, before vs after the cache-key work in CACHE_PLAN.md.
%
% Not benchmarked on the ernie head model (Scripts/ernie.m): far too slow to
% iterate on. Instead builds a 3-D cylinder model from EIDORS's own meshing
% tools (the recipe in Scripts/cylinder.m), sized so fmdl.R (dense
% n_sensors x n_elem, x3 for Rx/Ry/Rz) lands in the "few hundred MB" range
% where the cache-key pathology described in CACHE_PLAN.md is visible.
%
% The model is expensive to build (Netgen meshing + R quadrature, ~100s) and
% is cached to bench_fwd_solve_cache_fmdl.mat (gitignored) so repeat runs are
% fast; delete that file to force a rebuild.

fullpath      = mfilename('fullpath');
tests_folder  = fileparts(fullpath);
workspace_dir = fileparts(tests_folder);
addpath(genpath(fullfile(workspace_dir,'functions')));
setup_eidors(workspace_dir);

fmdl_cache_file = fullfile(tests_folder,'bench_fwd_solve_cache_fmdl.mat');
fmdl = build_or_load_cylinder(fmdl_cache_file);

n_elem    = size(fmdl.elems,1);
n_nodes   = size(fmdl.nodes,1);
n_sensors = numel(fmdl.sensors);
n_stim    = numel(fmdl.stimulation);
R_bytes   = 8 * n_sensors * n_elem * 3; % Rx,Ry,Rz

fprintf('\n== bench_fwd_solve_cache ==\n');
fprintf('cylinder model: n_elem=%d  n_nodes=%d  n_sensors=%d  n_stim=%d\n', ...
    n_elem, n_nodes, n_sensors, n_stim);
fprintf('fmdl.R (Rx+Ry+Rz, dense n_sensors x n_elem): %.0f MB\n', R_bytes/1024^2);

img = mk_image(fmdl, 1.0);

% Cache instrumentation: hit/miss/eviction messages and a cache size summary.
eidors_cache('debug_on', 'system_mat');
eidors_cache('debug_on', 'fwd_model_parameters');

eidors_cache('clear_all');
tic; d1 = fwd_solve_1st_order_mdeit(img); t1 = toc;   % cold
tic; d2 = fwd_solve_1st_order_mdeit(img); t2 = toc;   % should be cheaper: cached system_mat/fwd_model_parameters
tic; d3 = fwd_solve_1st_order_mdeit(img); t3 = toc;
fprintf('\ncold %.2fs | warm %.2fs | warm %.2fs\n', t1, t2, t3);
fprintf('meas identical: %d\n', isequal(d1.meas, d2.meas));

eidors_cache

bench_gamma_forming_vs_matrix_free(img);

end


function bench_gamma_forming_vs_matrix_free(img)
% Phase 3: forming Gamma explicitly (compute_gamma_matrices + Gamma*u) vs
% matrix-free (apply_gamma_matrices), on the same cylinder model/img. Uses
% the node solution v from a plain fwd_solve (skip_mag_field), so both paths
% see the identical u.

fprintf('\n== Gamma: forming vs matrix-free (same cylinder model) ==\n');

img.fwd_solve.get_all_meas = 1;
d = fwd_solve_1st_order_mdeit(img, true); % skip_mag_field: just need data.volt
u = d.volt(1:size(img.fwd_model.nodes,1),:);

n_sensors = numel(img.fwd_model.sensors);
n_nodes   = size(img.fwd_model.nodes,1);
n_stim    = size(u,2);

nrep = 5;
t_form = zeros(nrep,1);
for i = 1:nrep
    t0 = tic;
    imgg = compute_gamma_matrices(img);
    Bx = imgg.Gamma1*u; By = imgg.Gamma2*u; Bz = imgg.Gamma3*u;
    B_form = [Bx(:); By(:); Bz(:)];
    t_form(i) = toc(t0);
end

t_mf = zeros(nrep,1);
for i = 1:nrep
    t0 = tic;
    B_mf = apply_gamma_matrices(img, u);
    t_mf(i) = toc(t0);
end

rel = norm(B_mf(:) - B_form(:)) / norm(B_form(:));

gamma_c_bytes = 8*n_sensors*n_nodes*3 * 2; % Gamma{1,2,3} + Cx,Cy,Cz, each n_sensors x n_nodes
mf_bytes      = 8*n_sensors*n_stim*2;      % largest matrix-free temporary, [n_sensors x 2*n_stim]

fprintf('n_sensors=%d n_nodes=%d n_stim=%d\n', n_sensors, n_nodes, n_stim);
fprintf('forming Gamma : mean=%.3fs over %d reps\n', mean(t_form), nrep);
fprintf('matrix-free   : mean=%.3fs over %d reps\n', mean(t_mf), nrep);
fprintf('speedup (form/matrix-free) = %.2fx\n', mean(t_form)/mean(t_mf));
fprintf('largest dense temporary: forming ~%.1f MB (6x [n_sensors x n_nodes])  vs  matrix-free ~%.3f MB ([n_sensors x 2*n_stim])\n', ...
    gamma_c_bytes/1024^2, mf_bytes/1024^2);
fprintf('rel diff matrix-free vs forming = %.3e\n', rel);

end


function fmdl = build_or_load_cylinder(fmdl_cache_file)
if exist(fmdl_cache_file,'file')
    fprintf('Loading cached benchmark model from %s\n', fmdl_cache_file);
    temp = load(fmdl_cache_file);
    fmdl = temp.fmdl;
    return
end

fprintf('Building benchmark cylinder model (Netgen mesh + R quadrature, ~100s)...\n');

% Sized so n_elem is large enough that fmdl.R (dense n_sensors x n_elem) is a
% few hundred MB - the whole point is that the hashing/Gamma cost only
% becomes visible once R is substantial.
height = 3; radius = 1.0; maxsz = 0.06;
num_electrodes_ring = 16; num_rings = 4;
ring_vert_pos = height/(num_rings+1):height/(num_rings+1):height-height/(num_rings+1);
electrode_radius = 0.08;

elec_pos   = [num_electrodes_ring, ring_vert_pos];
elec_shape = [electrode_radius, electrode_radius, maxsz];

fmdl = ng_mk_cyl_models([height,radius,maxsz], elec_pos, elec_shape);
fmdl.system_mat  = @eidors_default;

num_electrodes = numel(fmdl.electrode);
fmdl.stimulation = mk_stim_patterns(num_electrodes,1,[0,1],[0,1],{'meas_current'},1);

% Magnetometers: one ring of sensors per electrode ring (mdeit3: 3 axes/sensor)
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
sensor_axes = struct('axis1',[1,0,0],'axis2',[0,1,0],'axis3',[0,0,1]);
fmdl = assign_magnetometers(fmdl, sensor_positions, sensor_axes);

mu0 = 1;
fmdl = compute_geometry_matrices(fmdl, mu0);

save(fmdl_cache_file, 'fmdl', '-v7.3');
end
