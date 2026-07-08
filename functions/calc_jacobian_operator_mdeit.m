function Jop = calc_jacobian_operator_mdeit(img, progress)
% CALC_JACOBIAN_OPERATOR_MDEIT  Matrix-free Jacobian operator for mDEIT.
%
%   Jop = calc_jacobian_operator_mdeit(img)
%   Jop = calc_jacobian_operator_mdeit(img, progress)
%
% PROGRESS (optional): when true, each J*v / J'*w matvec prints a live
% counter as it streams through the element blocks (useful for large
% meshes where a single matvec takes a while). Default false. It can also
% be enabled via img.fwd_model.jacobian_operator.progress = true.
%
% Returns a struct Jop with:
%   Jop.size            -> [n_meas, n_elem], same shape J would have
%   Jop.mtimes(v)        -> computes J*v          (v is n_elem x 1)
%   Jop.mtimes_transp(w) -> computes J'*w         (w is n_meas x 1)
%
% Neither J nor J'*J is ever assembled. The expensive one-off work
% (Gamma matrices, forward solve u, adjoint solve lambda, factorisation)
% is done ONCE when Jop is built; mtimes/mtimes_transp are then closures
% suitable for repeated calls inside an iterative solver (pcg, lsqr, ...).
%
% MEMORY: unlike a naive assembly, this operator never forms a dense
% [num_sensors x n_elem] array. The adjoint fields lambda are kept dense
% ([n_nodes x num_sensors], ~5.5x smaller than a [num_sensors x n_elem]
% factor for tetrahedral meshes), the gradient operators G.G{x,y,z} are
% kept sparse, and the element contraction inside mtimes/mtimes_transp is
% streamed in element blocks so a full [num_sensors x n_elem] temporary is
% never allocated. This makes the operator usable on fine meshes where
% the precomputed-factor form runs out of memory.
%
% The block size is chosen automatically to cap the size of the per-block
% dense temporaries. Override it (in elements) with:
%   img.fwd_model.jacobian_operator.chunk_size = 4000;
%
% Example - Gauss-Newton style normal-equations matvec with Tikhonov reg:
%   Jop = calc_jacobian_operator_mdeit(img);
%   afun = @(v) Jop.mtimes_transp(Jop.mtimes(v)) + alpha*v;
%   dsigma = pcg(afun, Jop.mtimes_transp(residual), tol, maxit);
%
% This mirrors calc_jacobian_mdeit.m (same recon_mode dispatch, same
% underlying math), re-organised for a matrix-free, memory-lean matvec.

if nargin < 2 || isempty(progress)
    progress = false;
    try
        progress = logical(img.fwd_model.jacobian_operator.progress);
    end
end

recon_mode = check_recon_mode(img);

switch recon_mode
    case 'mdeit1'
        Jop = setup_operator_1axis(img, progress);
    case 'mdeit3'
        Jop = setup_operator_3axis(img, progress);
end

end


%% ------------------------------------------------------------------
function Jop = setup_operator_1axis(img, progress)

mu0 = img.fwd_model.mu0;

n_nodes = size(img.fwd_model.nodes,1);
n_elem  = size(img.fwd_model.elems,1);

num_stim       = numel(img.fwd_model.stimulation);
num_sensors    = numel(img.fwd_model.sensors);
num_electrodes = numel(img.fwd_model.electrode);

% Compute Gamma matrices (same as calc_jacobian_1axis)
img = compute_gamma_matrices(img);

R = img.fwd_model.R;
G = img.fwd_model.G;
Gamma = img.Gamma;

% Factorise lhs system matrix once
A_matrix = lhs_eit_full(img);
F = factorise_symmetric(A_matrix);

% Forward solve u for all stim patterns
I = zeros(num_electrodes,num_stim);
for j = 1:num_stim
    I(:,j) = img.fwd_model.stimulation(j).stim_pattern;
end
rhs = sparse(n_nodes+num_electrodes,num_stim);
rhs(end-num_electrodes+1:end,:) = I;
u = solve_fact_multiple_rhs(F,rhs);
u = u(1:end-num_electrodes,:);

% Adjoint solve lambda for all sensors  ([n_nodes x num_sensors], dense)
GammaT = Gamma.';
rhs = sparse(n_nodes+num_electrodes,num_sensors);
rhs(1:end-num_electrodes,:) = -GammaT;
lambda = solve_fact_multiple_rhs(F,rhs);
lambda = lambda(1:end-num_electrodes,:);

% Small dense u-gradient factors ([n_elem x num_stim], num_stim is small).
Gxu = G.Gx*u;  Gyu = G.Gy*u;  Gzu = G.Gz*u;

Rx = R.Rx; Ry = R.Ry; Rz = R.Rz;      % sparse [num_sensors x n_elem]
elemV = img.fwd_model.elem_volume(:); % [n_elem x 1]

g = zeros(num_sensors,3);
for m = 1:num_sensors
    g(m,:) = img.fwd_model.sensors(m).axes.axis;
end
gx = g(:,1); gy = g(:,2); gz = g(:,3);

mu_factor = mu0/(4*pi);

chunk_size = get_chunk_size(img, num_sensors, n_elem);
chunks = make_chunks(n_elem, chunk_size);
nchunks = size(chunks,1);

Jop.size = [num_sensors*num_stim, n_elem];

Jop.mtimes = @(v) reshape( ...
    block_forward_matvec(v, G, lambda, Gxu,Gyu,Gzu, Rx,Ry,Rz, ...
        elemV, gx,gy,gz, mu_factor, chunks, ...
        make_progress(progress, nchunks, 'J*v')), [], 1);

Jop.mtimes_transp = @(w) block_transp_matvec( ...
    reshape(w, num_sensors, num_stim), G, lambda, Gxu,Gyu,Gzu, ...
    Rx,Ry,Rz, elemV, gx,gy,gz, mu_factor, chunks, n_elem, ...
    make_progress(progress, nchunks, 'J''*w'));

end


%% ------------------------------------------------------------------
function Jop = setup_operator_3axis(img, progress)

mu0 = img.fwd_model.mu0;
n_elem = size(img.fwd_model.elems,1);

num_stim    = numel(img.fwd_model.stimulation);
num_sensors = numel(img.fwd_model.sensors);

img = compute_gamma_matrices(img);
R = img.fwd_model.R;
G = img.fwd_model.G;
Gamma1 = img.Gamma1; Gamma2 = img.Gamma2; Gamma3 = img.Gamma3;

% Forward solve u (shared across all 3 blocks)
img.fwd_solve.get_all_meas = 1;
u = fwd_solve(img);
u = u.volt;

% Build reduced (electrode-eliminated) system matrix, same as
% calc_jacobian_3axis, and factorise once for the 3 adjoint solves
A_matrix = lhs_eit_full(img);
n_nodes = size(img.fwd_model.nodes,1);
n_elec  = numel(img.fwd_model.electrode);

Ac  = A_matrix(1:n_nodes,1:n_nodes);
Ae  = A_matrix(1:n_nodes,n_nodes+1:end);
Aet = A_matrix(n_nodes+1:end,1:n_nodes);
Ad  = A_matrix(n_nodes+1:end,n_nodes+1:end);
Ad1 = sparse(1:n_elec,1:n_elec,1./diag(Ad),n_elec,n_elec);

A_reduced = Ac - Ae*Ad1*Aet;
F = factorise_symmetric(A_reduced);

% Adjoint fields, kept dense ([n_nodes x num_sensors] each)
lambdaX = solve_fact_multiple_rhs(F, -Gamma1.');
lambdaY = solve_fact_multiple_rhs(F, -Gamma2.');
lambdaZ = solve_fact_multiple_rhs(F, -Gamma3.');

% Small dense u-gradient factors ([n_elem x num_stim], num_stim is small)
Gxu = G.Gx*u;  Gyu = G.Gy*u;  Gzu = G.Gz*u;

Rx = R.Rx; Ry = R.Ry; Rz = R.Rz;      % sparse [num_sensors x n_elem]
elemV = img.fwd_model.elem_volume(:);
mu_factor = mu0/(4*pi);

g = zeros(num_sensors,3,3);
for m = 1:num_sensors
    g(m,:,:) = [ ...
        img.fwd_model.sensors(m).axes.axis1; ...
        img.fwd_model.sensors(m).axes.axis2; ...
        img.fwd_model.sensors(m).axes.axis3];
end

lambdas = {lambdaX, lambdaY, lambdaZ};
blocks = struct('lambda',{},'gx',{},'gy',{},'gz',{});
for k = 1:3
    blocks(k).lambda = lambdas{k};
    blocks(k).gx = g(:,k,1);
    blocks(k).gy = g(:,k,2);
    blocks(k).gz = g(:,k,3);
end

chunk_size = get_chunk_size(img, num_sensors, n_elem);
chunks = make_chunks(n_elem, chunk_size);
nchunks = size(chunks,1);

nrow = num_sensors*num_stim;
Jop.size = [3*nrow, n_elem];

Jop.mtimes        = @mtimes_3axis;
Jop.mtimes_transp = @mtimes_transp_3axis;

    function Jv = mtimes_3axis(v)
        Jv = zeros(3*nrow,1);
        tick = make_progress(progress, 3*nchunks, 'J*v');
        for k = 1:3
            Mk = block_forward_matvec(v, G, blocks(k).lambda, ...
                Gxu,Gyu,Gzu, Rx,Ry,Rz, elemV, ...
                blocks(k).gx, blocks(k).gy, blocks(k).gz, ...
                mu_factor, chunks, tick);
            Jv((k-1)*nrow+1:k*nrow) = Mk(:);
        end
    end

    function JtW = mtimes_transp_3axis(w)
        JtW = zeros(n_elem,1);
        tick = make_progress(progress, 3*nchunks, 'J''*w');
        for k = 1:3
            Wk = reshape(w((k-1)*nrow+1:k*nrow), num_sensors, num_stim);
            JtW = JtW + block_transp_matvec(Wk, G, blocks(k).lambda, ...
                Gxu,Gyu,Gzu, Rx,Ry,Rz, elemV, ...
                blocks(k).gx, blocks(k).gy, blocks(k).gz, ...
                mu_factor, chunks, n_elem, tick);
        end
    end

end


%% ==================== CORE CONTRACTION KERNELS =====================
% Same math as calc_jacobian_mdeit, but the contraction over elements is
% streamed in blocks. For each block c we form only the small dense
% [num_sensors x numel(c)] factor grad(lambda)|_c on the fly from the
% sparse G.G{x,y,z} and the dense lambda, so no [num_sensors x n_elem]
% array is ever allocated.

function M = block_forward_matvec(v, G, lambda, Gxu,Gyu,Gzu, ...
                                  Rx,Ry,Rz, elemV, gx,gy,gz, mu_factor, chunks, tick)
% Computes M = J_block * v as a [num_sensors x num_stim] matrix
% (vectorize with M(:) to match calc_jacobian_mdeit's row ordering).
%
%   v      : [n_elem x 1]              perturbation direction
%   lambda : [n_nodes x num_sensors]   adjoint field (dense)
%   Gxu... : [n_elem x num_stim]       G.G{x,y,z}*u
%   Rx,Ry,Rz: [num_sensors x n_elem]   (sparse)
%   elemV  : [n_elem x 1]
%   gx,gy,gz: [num_sensors x 1]        sensor-axis components for this block

Gx = G.Gx; Gy = G.Gy; Gz = G.Gz;
num_sensors = size(lambda,2);
num_stim    = size(Gxu,2);
M = zeros(num_sensors, num_stim);

for ci = 1:size(chunks,1)
    c = chunks(ci,1):chunks(ci,2);

    % grad(lambda) on this block, [num_sensors x |c|]
    GxLc = (Gx(c,:)*lambda).';
    GyLc = (Gy(c,:)*lambda).';
    GzLc = (Gz(c,:)*lambda).';

    Gxuc = Gxu(c,:);  Gyuc = Gyu(c,:);  Gzuc = Gzu(c,:);  % [|c| x num_stim]

    % --- conductivity term: elemV * (grad(lambda) . grad(u)) ---
    vEc = (v(c).' .* elemV(c).');       % [1 x |c|]
    M = M + (GxLc.*vEc)*Gxuc + (GyLc.*vEc)*Gyuc + (GzLc.*vEc)*Gzuc;

    % --- position term: mu_factor * g . (R x grad(u)) ---
    vRc = v(c).';                       % [1 x |c|]
    Rxc = Rx(:,c).*vRc;  Ryc = Ry(:,c).*vRc;  Rzc = Rz(:,c).*vRc;

    dCxdp = -Rzc*Gyuc + Ryc*Gzuc;
    dCydp = -Rxc*Gzuc + Rzc*Gxuc;
    dCzdp = -Ryc*Gxuc + Rxc*Gyuc;

    M = M + mu_factor*( gx.*dCxdp + gy.*dCydp + gz.*dCzdp );

    if ~isempty(tick), tick(); end
end

end


function v_out = block_transp_matvec(W, G, lambda, Gxu,Gyu,Gzu, ...
                                     Rx,Ry,Rz, elemV, gx,gy,gz, mu_factor, chunks, n_elem, tick)
% Computes v_out = J_block' * w, where W = reshape(w,num_sensors,num_stim).
% Returns v_out as [n_elem x 1].

Gx = G.Gx; Gy = G.Gy; Gz = G.Gz;
v_out = zeros(n_elem,1);

Wg1 = W.*gx;  Wg2 = W.*gy;  Wg3 = W.*gz;   % [num_sensors x num_stim]

for ci = 1:size(chunks,1)
    c = chunks(ci,1):chunks(ci,2);

    Gxuc = Gxu(c,:);  Gyuc = Gyu(c,:);  Gzuc = Gzu(c,:);  % [|c| x num_stim]

    % --- conductivity term ---
    WUxc = W*Gxuc.';  WUyc = W*Gyuc.';  WUzc = W*Gzuc.';   % [num_sensors x |c|]

    GxLc = (Gx(c,:)*lambda).';
    GyLc = (Gy(c,:)*lambda).';
    GzLc = (Gz(c,:)*lambda).';

    dfdx_tc = elemV(c).' .* sum(GxLc.*WUxc + GyLc.*WUyc + GzLc.*WUzc, 1); % [1 x |c|]

    % --- position term ---
    A1c = Wg1*Gyuc.';  B1c = Wg1*Gzuc.';
    A2c = Wg2*Gzuc.';  B2c = Wg2*Gxuc.';
    A3c = Wg3*Gxuc.';  B3c = Wg3*Gyuc.';

    Rxc = Rx(:,c);  Ryc = Ry(:,c);  Rzc = Rz(:,c);

    term1 = -sum(Rzc.*A1c,1) + sum(Ryc.*B1c,1);
    term2 = -sum(Rxc.*A2c,1) + sum(Rzc.*B2c,1);
    term3 = -sum(Ryc.*A3c,1) + sum(Rxc.*B3c,1);

    dfdp_tc = mu_factor*(term1 + term2 + term3);

    v_out(c) = (dfdx_tc + dfdp_tc).';   % [|c| x 1]

    if ~isempty(tick), tick(); end
end

end


%% ==================== BLOCKING HELPERS =============================

function chunk_size = get_chunk_size(img, num_sensors, n_elem)
% Element block size. Caps the per-block dense temporary
% (~[num_sensors x chunk_size]) at ~64 MB unless overridden.
chunk_size = [];
try
    chunk_size = img.fwd_model.jacobian_operator.chunk_size;
end
if isempty(chunk_size)
    budget_bytes = 64e6;
    chunk_size = floor(budget_bytes / (8*max(num_sensors,1)));
end
chunk_size = max(1, min(chunk_size, n_elem));
end

function chunks = make_chunks(n_elem, chunk_size)
starts = (1:chunk_size:n_elem).';
ends   = min(starts + chunk_size - 1, n_elem);
chunks = [starts ends];
end

function tick = make_progress(on, total, label)
% Returns a per-chunk progress callback (or [] when disabled). Each call
% advances a live \r counter; a newline + elapsed time is printed on the
% final chunk. Matches the fprintf progress style used elsewhere in the
% mDEIT solvers.
if ~on
    tick = [];
    return
end
count = 0;
t0 = tic;
tick = @do_tick;
    function do_tick()
        count = count + 1;
        fprintf('\r  Jop %s: block %d/%d (%3.0f%%)', ...
                label, count, total, 100*count/total);
        if count >= total
            fprintf('  [%.1fs]\n', toc(t0));
        end
    end
end


%% ==================== SHARED HELPERS (as in calc_jacobian_mdeit) ===

function F = factorise_symmetric(A)
F.kind = 'ldl';
try
    [F.L,F.D,F.P] = ldl(A,'vector');
    F.n = size(A,1);
catch
    error('Couldnt do it')
end
end

function X = solve_fact_multiple_rhs(F, rhs)
switch F.kind
    case 'ldl'
        rp = rhs(F.P, :);
        y  = F.L \ rp;
        z  = F.D \ y;
        w  = F.L' \ z;
        X = zeros(F.n, size(rhs,2));
        X(F.P, :) = w;
    otherwise
        error('Unknown factorisation kind.');
end
end

function recon_mode = check_recon_mode(img)
if isfield(img.fwd_model.sensors(1).axes, 'axis')
    recon_mode = 'mdeit1';
elseif isfield(img.fwd_model.sensors(1).axes, 'axis1') && ...
        isfield(img.fwd_model.sensors(1).axes, 'axis2') && ...
        isfield(img.fwd_model.sensors(1).axes, 'axis3')
    recon_mode = 'mdeit3';
else
    error('Unknown sensor axes configuration. Please check the sensor axes fields.');
end
end
