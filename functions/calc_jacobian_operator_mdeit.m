function Jop = calc_jacobian_operator_mdeit(img)
% CALC_JACOBIAN_OPERATOR_MDEIT  Matrix-free Jacobian operator for mDEIT.
%
%   Jop = calc_jacobian_operator_mdeit(img)
%
% Returns a struct Jop with:
%   Jop.size            -> [n_meas, n_elem], same shape J would have
%   Jop.mtimes(v)        -> computes J*v          (v is n_elem x 1)
%   Jop.mtimes_transp(w) -> computes J'*w         (w is n_meas x 1)
%
% Neither J nor J'*J is ever assembled. All the expensive one-off work
% (Gamma matrices, forward solve u, adjoint solve lambda, factorisation)
% is done ONCE when Jop is built; mtimes/mtimes_transp are then cheap
% closures suitable for repeated calls inside an iterative solver
% (pcg, lsqr, your own CG on the normal equations, etc).
%
% Example - Gauss-Newton style normal-equations matvec with Tikhonov reg:
%   Jop = calc_jacobian_operator_mdeit(img);
%   afun = @(v) Jop.mtimes_transp(Jop.mtimes(v)) + alpha*v;
%   dsigma = pcg(afun, Jop.mtimes_transp(residual), tol, maxit);
%
% This mirrors calc_jacobian_mdeit.m (same recon_mode dispatch, same
% underlying math), just re-organised so the element contraction is done
% via 2-D matrix products instead of broadcasting a 3-D array.

recon_mode = check_recon_mode(img);

switch recon_mode
    case 'mdeit1'
        Jop = setup_operator_1axis(img);
    case 'mdeit3'
        Jop = setup_operator_3axis(img);
end

end


%% ------------------------------------------------------------------
function Jop = setup_operator_1axis(img)

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

% Adjoint solve lambda for all sensors
GammaT = Gamma.';
rhs = sparse(n_nodes+num_electrodes,num_sensors);
rhs(1:end-num_electrodes,:) = -GammaT;
lambda = solve_fact_multiple_rhs(F,rhs);
lambda = lambda(1:end-num_electrodes,:);

% Pre-form the 2-D gradient factors (sensor x elem, stim x elem)
GxL = (G.Gx*lambda).';  GyL = (G.Gy*lambda).';  GzL = (G.Gz*lambda).';
GxU = (G.Gx*u).';       GyU = (G.Gy*u).';       GzU = (G.Gz*u).';

Rx = R.Rx; Ry = R.Ry; Rz = R.Rz;      % [num_sensors x n_elem]
elemV = img.fwd_model.elem_volume(:); % [n_elem x 1]

g = zeros(num_sensors,3);
for m = 1:num_sensors
    g(m,:) = img.fwd_model.sensors(m).axes.axis;
end
gx = g(:,1); gy = g(:,2); gz = g(:,3);

mu_factor = mu0/(4*pi);

Jop.size = [num_sensors*num_stim, n_elem];

Jop.mtimes = @(v) reshape( ...
    block_forward_matvec(v, GxL,GyL,GzL, GxU,GyU,GzU, Rx,Ry,Rz, ...
                          elemV, gx,gy,gz, mu_factor), [], 1);

Jop.mtimes_transp = @(w) block_transp_matvec( ...
    reshape(w, num_sensors, num_stim), ...
    GxL,GyL,GzL, GxU,GyU,GzU, Rx,Ry,Rz, elemV, gx,gy,gz, mu_factor);

end


%% ------------------------------------------------------------------
function Jop = setup_operator_3axis(img)

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

lambdaX = solve_fact_multiple_rhs(F, -Gamma1.');
lambdaY = solve_fact_multiple_rhs(F, -Gamma2.');
lambdaZ = solve_fact_multiple_rhs(F, -Gamma3.');

GxU = (G.Gx*u).'; GyU = (G.Gy*u).'; GzU = (G.Gz*u).';
Rx = R.Rx; Ry = R.Ry; Rz = R.Rz;
elemV = img.fwd_model.elem_volume(:);
mu_factor = mu0/(4*pi);

GxLX=(G.Gx*lambdaX).'; GyLX=(G.Gy*lambdaX).'; GzLX=(G.Gz*lambdaX).';
GxLY=(G.Gx*lambdaY).'; GyLY=(G.Gy*lambdaY).'; GzLY=(G.Gz*lambdaY).';
GxLZ=(G.Gx*lambdaZ).'; GyLZ=(G.Gy*lambdaZ).'; GzLZ=(G.Gz*lambdaZ).';

g = zeros(num_sensors,3,3);
for m = 1:num_sensors
    g(m,:,:) = [ ...
        img.fwd_model.sensors(m).axes.axis1; ...
        img.fwd_model.sensors(m).axes.axis2; ...
        img.fwd_model.sensors(m).axes.axis3];
end

GxL_all = {GxLX,GxLY,GxLZ};
GyL_all = {GyLX,GyLY,GyLZ};
GzL_all = {GzLX,GzLY,GzLZ};

blocks = struct('GxL',{},'GyL',{},'GzL',{},'gx',{},'gy',{},'gz',{});
for k = 1:3
    blocks(k).GxL = GxL_all{k};
    blocks(k).GyL = GyL_all{k};
    blocks(k).GzL = GzL_all{k};
    blocks(k).gx  = g(:,k,1);
    blocks(k).gy  = g(:,k,2);
    blocks(k).gz  = g(:,k,3);
end

nrow = num_sensors*num_stim;
Jop.size = [3*nrow, n_elem];

Jop.mtimes        = @mtimes_3axis;
Jop.mtimes_transp = @mtimes_transp_3axis;

    function Jv = mtimes_3axis(v)
        Jv = zeros(3*nrow,1);
        for k = 1:3
            Mk = block_forward_matvec(v, ...
                blocks(k).GxL, blocks(k).GyL, blocks(k).GzL, ...
                GxU,GyU,GzU, Rx,Ry,Rz, elemV, ...
                blocks(k).gx, blocks(k).gy, blocks(k).gz, mu_factor);
            Jv((k-1)*nrow+1:k*nrow) = Mk(:);
        end
    end

    function JtW = mtimes_transp_3axis(w)
        JtW = zeros(n_elem,1);
        for k = 1:3
            Wk = reshape(w((k-1)*nrow+1:k*nrow), num_sensors, num_stim);
            JtW = JtW + block_transp_matvec(Wk, ...
                blocks(k).GxL, blocks(k).GyL, blocks(k).GzL, ...
                GxU,GyU,GzU, Rx,Ry,Rz, elemV, ...
                blocks(k).gx, blocks(k).gy, blocks(k).gz, mu_factor);
        end
    end

end


%% ==================== CORE CONTRACTION KERNELS =====================
% These two functions are the whole trick: instead of forming a
% [num_sensors x num_stim x n_elem] array and multiplying by v (or w)
% along the elem dimension, we contract over elements using 2-D
% matrix products. Same FLOP count as one Jacobian assembly, but no
% 3-D array (and no J, no J'*J) is ever formed.

function M = block_forward_matvec(v, GxL,GyL,GzL, GxU,GyU,GzU, ...
                                   Rx,Ry,Rz, elemV, gx,gy,gz, mu_factor)
% Computes M = J_block * v as a [num_sensors x num_stim] matrix
% (vectorize with M(:) to match calc_jacobian_mdeit's row ordering).
%
%   v      : [n_elem x 1]   perturbation direction
%   GxL... : [num_sensors x n_elem]   G.G{x,y,z}*lambda, transposed
%   GxU... : [num_stim    x n_elem]   G.G{x,y,z}*u,      transposed
%   Rx,Ry,Rz: [num_sensors x n_elem]
%   elemV  : [n_elem x 1]
%   gx,gy,gz: [num_sensors x 1]  sensor-axis components for this block

vE = (v(:).' .* elemV(:).');   % element weight incl. volume, for dfdx
vR =  v(:).';                 % element weight, no volume,   for dfdp

% --- conductivity term: elemV * (grad(lambda) . grad(u)) ---
M = (GxL.*vE)*GxU.' + (GyL.*vE)*GyU.' + (GzL.*vE)*GzU.';

% --- position term: mu_factor * g . (R x grad(u)) ---
Rx_v = Rx.*vR;  Ry_v = Ry.*vR;  Rz_v = Rz.*vR;

dCxdp = -Rz_v*GyU.' + Ry_v*GzU.';
dCydp = -Rx_v*GzU.' + Rz_v*GxU.';
dCzdp = -Ry_v*GxU.' + Rx_v*GyU.';

M = M + mu_factor*( gx.*dCxdp + gy.*dCydp + gz.*dCzdp );

end


function v_out = block_transp_matvec(W, GxL,GyL,GzL, GxU,GyU,GzU, ...
                                      Rx,Ry,Rz, elemV, gx,gy,gz, mu_factor)
% Computes v_out = J_block' * w, where W = reshape(w,num_sensors,num_stim).
% Returns v_out as [n_elem x 1].

% --- conductivity term ---
WUx = W*GxU;  WUy = W*GyU;  WUz = W*GzU;   % [num_sensors x n_elem]

dfdx_t = elemV(:).' .* sum(GxL.*WUx + GyL.*WUy + GzL.*WUz, 1); % [1 x n_elem]

% --- position term ---
Wg1 = W.*gx;  Wg2 = W.*gy;  Wg3 = W.*gz;   % [num_sensors x num_stim]

A1 = Wg1*GyU;  B1 = Wg1*GzU;
A2 = Wg2*GzU;  B2 = Wg2*GxU;
A3 = Wg3*GxU;  B3 = Wg3*GyU;

term1 = -sum(Rz.*A1,1) + sum(Ry.*B1,1);
term2 = -sum(Rx.*A2,1) + sum(Rz.*B2,1);
term3 = -sum(Ry.*A3,1) + sum(Rx.*B3,1);

dfdp_t = mu_factor*(term1 + term2 + term3);

v_out = (dfdx_t + dfdp_t).';   % [n_elem x 1]

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
