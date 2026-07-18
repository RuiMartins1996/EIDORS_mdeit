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

bench_pcg_ichol_cache(fmdl);

end


function bench_pcg_ichol_cache(fmdl)
% Phase 4 (ICHOL_CACHE_PLAN.md): the ichol preconditioner used by the pcg
% solve path, cached via eidors_cache keyed on the reduced matrix E. Same
% cylinder model as the rest of this file, with fwd_model.solve_mdeit.method
% = 'pcg' (the cached fmdl above defaults to left_divide, so this is a
% genuinely new configuration).

fprintf('\n== pcg path: ichol preconditioner cache (Phase 4) ==\n');

fmdl_pcg = fmdl;
fmdl_pcg.solve_mdeit.method = 'pcg';
img_pcg = mk_image(fmdl_pcg, 1.0);

eidors_cache('clear_all');
eidors_cache('debug_on', 'build_ichol');

tic; d1 = fwd_solve_1st_order_mdeit(img_pcg); t1 = toc;   % cold: builds ichol
tic; d2 = fwd_solve_1st_order_mdeit(img_pcg); t2 = toc;   % warm: reuses cached ichol
tic; d3 = fwd_solve_1st_order_mdeit(img_pcg); t3 = toc;

eidors_cache('debug_off', 'build_ichol');

fprintf('pcg path: cold %.2fs | warm %.2fs | warm %.2fs\n', t1, t2, t3);
fprintf('meas identical: %d\n', isequal(d1.meas, d2.meas));
fprintf(['NOTE: this cold/warm gap is NOT a clean build_ichol number - the ' ...
   'first parfor call in this MATLAB session also pays one-time ' ...
   'parallel-pool startup (tens of seconds; see the isolated timing ' ...
   'below for the real build_ichol figure).\n']);

% Isolate build_ichol alone, the same way the Phase 1 work isolated
% calc_system_mat's hash cost from the full solve. build_ichol is a LOCAL
% function of fwd_solve_1st_order_mdeit.m (deliberately not moved to its own
% file - see ICHOL_CACHE_PLAN.md), so it cannot be called or handled (@build_
% ichol) from this file: MATLAB local/subfunctions are only visible within
% their own defining file. Two consequences below:
%  1. The "cached" loop can still legitimately probe the real cache entry
%     that fwd_solve_1st_order_mdeit just populated above: eidors_cache's
%     shorthand interface looks up entries by (fstr, hash(cache_obj)) BEFORE
%     ever touching the function handle (see cache_shorthand in
%     eidors_cache.m) - the handle is only invoked on a genuine miss. So
%     passing any valid handle with the same fstr/cache_obj is a real cache
%     hit against the actual Lic build by the actual build_ichol, not a
%     simulation.
%  2. The "uncached" loop has no such option - there is no way to invoke the
%     real build_ichol from outside its file - so it runs LOCAL_BUILD_ICHOL
%     below, a literal copy of build_ichol's body (fwd_solve_1st_order_mdeit.m
%     lines 361-388), kept only for this benchmark's isolation number.
E = pcg_reduced_matrix(img_pcg);

nrep = 5;
copt.fstr = 'build_ichol';

t_cached = zeros(nrep,1);
for i = 1:nrep
   t0 = tic;
   Lic = eidors_cache(@local_build_ichol, {E}, copt); %#ok<NASGU> % real cache hit
   t_cached(i) = toc(t0);
end

t_uncached = zeros(nrep,1);
for i = 1:nrep
   t0 = tic;
   evalc('local_build_ichol(E);'); % evalc suppresses the per-rep fprintf
   t_uncached(i) = toc(t0);
end

fprintf('build_ichol alone: cached (cache hit) mean=%.4fs | uncached (rebuild) mean=%.4fs over %d reps\n', ...
   mean(t_cached), mean(t_uncached), nrep);
fprintf('speedup (uncached/cached) = %.1fx\n', mean(t_uncached)/mean(t_cached));

end


function E = pcg_reduced_matrix(img)
% Reproduce the exact reduced matrix E that fwd_solve_1st_order_mdeit builds
% internally for this img, so the isolated build_ichol timing above runs on
% the real E (and therefore hits the real cache entry). Mirrors
% fwd_solve_1st_order_mdeit.m's own lean-img system-matrix assembly
% (mdeit_lean_img + calc_system_mat) and its set_gnd_node fallback (pick the
% node nearest the model centroid) for the single-ground-node case that
% 'meas_current' stimulation always collapses to.
img_lean  = mdeit_lean_img(img);
fmdl_lean = img_lean.fwd_model;

s_mat = calc_system_mat(img_lean);

ctr = mean(fmdl_lean.nodes,1);
d2  = sum(bsxfun(@minus,fmdl_lean.nodes,ctr).^2,2);
[~,gnd_node] = min(d2);

idx = 1:size(s_mat.E,1);
idx(gnd_node) = [];
E = s_mat.E(idx,idx);
end


function Lic = local_build_ichol(E)
% Literal copy of fwd_solve_1st_order_mdeit.m's local BUILD_ICHOL, kept only
% to time an "uncached" rebuild from this file (see bench_pcg_ichol_cache) -
% real build_ichol cannot be called from outside its defining file. Keep in
% sync with the original by hand if that function changes.
base_opts = struct('type','ict','droptol',1e-3,'michol','on');
alpha = 0;
fprintf('local_build_ichol: Building iChol preconditioner\n');
for attempt = 1:8
   opts = base_opts; opts.diagcomp = alpha;
   try
      Lic = ichol(E, opts);
      return
   catch
      if alpha == 0
         dg = full(diag(E));
         alpha = max( max(sum(abs(E),2)./max(dg,eps)) - 2, 1e-4 );
      else
         alpha = alpha * 2;
      end
   end
end
eidors_msg(['local_build_ichol: ichol preconditioner failed; ' ...
   'falling back to unpreconditioned pcg'], 1);
Lic = [];
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
