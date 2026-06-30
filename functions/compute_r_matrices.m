function [Rx,Ry,Rz] = compute_r_matrices(fmdl,quad_mode)

% Implemented quadrature rules:
% Check: https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html

% 3D:
% nco6: Newton-Cotes Open (6)
% nco2: Newton-Cotes Open (2)
% nco0: Newton-Cotes Open (0) / Centroid quadrature

% 2D:
% toms37: ACM TOMS algorithm #706 (TOMS706_37)
% centroid: centroid quadrature for triangle

% Check fmdl
assert(isfield(fmdl,"sensors"),'fmdl must have field "sensors"');
assert(isfield(fmdl,"nodes"),'fmdl must have field "nodes"');

ndims = size(fmdl.nodes,2);

assert(ndims == 2 || ndims == 3,'Node dimensions must be either 2D or 3D');

if nargin<2
    if ndims == 2
        quad_mode = 'toms37';
    else 
        quad_mode = 'nco6';
    end
end

% Assert quad_mode is valid
check_valid_quadrature_rule(quad_mode,ndims);


if ndims ==3
    if size(fmdl.elems,1) > 30000
        [Rx,Ry,Rz] = compute_r_matrices_quadrature_3d_parallelization_over_elements(fmdl,quad_mode);
    else
        [Rx,Ry,Rz] = compute_r_matrices_quadrature_3d(fmdl,quad_mode);
    end
else
    if size(fmdl.elems,1) > 30000
        [Rx,Ry,Rz] = compute_r_matrices_quadrature_2d_parallelization_over_elements(fmdl,quad_mode);
    else
        [Rx,Ry,Rz] = compute_r_matrices_quadrature_2d(fmdl,quad_mode);
    end
end

end


function check_valid_quadrature_rule(quad_mode,ndims)

implemented_quad_3d = ["nco6","nco2","nco0"];
implemented_quad_2d = ["toms37","centroid"];

if ndims == 3

    if ~ismember(quad_mode, implemented_quad_3d)
        error(['Invalid 3D quadrature rule. Available rules: %s'], ...
            strjoin(implemented_quad_3d, ', '));
    end

elseif ndims == 2

    if ~ismember(quad_mode, implemented_quad_2d)
        error(['Invalid 2D quadrature rule. Available rules: %s'], ...
            strjoin(implemented_quad_2d, ', '));
    end

else

    error('fmdl.nodes must be either 2D or 3D');

end

end


% Old strategy:
% Parallelization over number of sensors: each worker has a huge array
% broadcast into memory. Old approach is faster when elem_p is small, but
% when it gets very large, the price of broadcasting is huge!

% New strategy:
% Parallelization over number of elements: each worker has a small array
% broadcast into memory



%% FUNCTION: get_quadrature_3d
function [quad_l,quad_w,nq] = get_quadrature_3d(quad_mode)
switch quad_mode
    case 'nco6'
        [quad_l,quad_w,nq] = get_newton_cotes_open_6();
    case 'nco2'
        [quad_l,quad_w,nq] = get_newton_cotes_open_2();
    case 'nco0'
        [quad_l,quad_w,nq] = get_newton_cotes_open_0();
    otherwise
        error('Unexpected');
end
end

%% FUNCTION: get_quadrature_2d
function [quad_l,quad_w,nq] = get_quadrature_2d(quad_mode)
switch quad_mode
    case 'toms37'
        [quad_l,quad_w,nq] = get_toms_37();
    case 'centroid'
        [quad_l,quad_w,nq] = get_centroid_quadrature();
    otherwise
        error('Unexpected');
end
end


%% FUNCITON: compute_r_matrices_quadrature_3d_parallelization_over_elements
function [Rx,Ry,Rz] = compute_r_matrices_quadrature_3d_parallelization_over_elements(fmdl,quad_mode)

nodes    = fmdl.nodes;      % N×3
elements = fmdl.elems;      % Ntets×4
sensors  = fmdl.sensors;
n_elem    = size(elements,1);
n_sensors = length(sensors);

Rx = zeros(n_sensors,n_elem);
Ry = zeros(n_sensors,n_elem);
Rz = zeros(n_sensors,n_elem);

persistent quad_l quad_w nq;

[quad_l,quad_w,nq] = get_quadrature_3d(quad_mode);

parfor k = 1:n_elem
    v = nodes(elements(k,:),:);

    % Geometry
    A  = v(1,:);
    BA = (v(2,:) - A).';
    CA = (v(3,:) - A).';
    DA = (v(4,:) - A).';

    J = abs(det([BA CA DA])) / 6;

    % Quadrature (local, small)
    l2 = quad_l(:,1);
    l3 = quad_l(:,2);
    l4 = quad_l(:,3);
    l1 = 1 - l2 - l3 - l4;

    qp = l1*A + l2*v(2,:) + l3*v(3,:) + l4*v(4,:);  % [nq × 3]

    % Loop over sensors (small dimension)
    for m = 1:n_sensors
        c = sensors(m).position(:).';

        dx = c(1) - qp(:,1);
        dy = c(2) - qp(:,2);
        dz = c(3) - qp(:,3);

        r2 = dx.^2 + dy.^2 + dz.^2;
        r3 = r2 .* sqrt(r2);

        Ix = (quad_w.' * (dx ./ r3)) * J;
        Iy = (quad_w.' * (dy ./ r3)) * J;
        Iz = (quad_w.' * (dz ./ r3)) * J;

        Rx(m,k) = Ix;
        Ry(m,k) = Iy;
        Rz(m,k) = Iz;
    end
end

end

%% FUNCTION: compute_r_matrices_quadrature_2d_parallelization_over_elements
function [Rx,Ry,Rz] = compute_r_matrices_quadrature_2d_parallelization_over_elements(fmdl,quad_mode)

nodes    = fmdl.nodes;
elements = fmdl.elems;
sensors  = fmdl.sensors;

n_elem    = size(elements,1);
n_sensors = length(sensors);

Rx = zeros(n_sensors,n_elem);
Ry = zeros(n_sensors,n_elem);
Rz = zeros(n_sensors,n_elem);

persistent quad_l quad_w nq;
[quad_l,quad_w,nq] = get_quadrature_2d(quad_mode);

% Parallel over elements (key change)
parfor k = 1:n_elem

    % --- Element geometry ---
    v = nodes(elements(k,:),:);    % [A; B; C]
    A = v(1,:); 
    B = v(2,:); 
    C = v(3,:);

    % Jacobian (area)
    J = 0.5 * abs(det([B - A; C - A]));

    % --- Quadrature points (local, small) ---
    xi  = quad_l(:,1);
    eta = quad_l(:,2);

    % Vectorized mapping: [nq × 2]
    qp = A + xi.*(B - A) + eta.*(C - A);

    % --- Loop over sensors (small dimension) ---
    for m = 1:n_sensors
        c = sensors(m).position(:).';

        dx = c(1) - qp(:,1);
        dy = c(2) - qp(:,2);

        % planar geometry: z handled explicitly
        dz = c(3);  % quadrature points are at z = 0

        r2 = dx.^2 + dy.^2 + dz.^2;
        r3 = r2 .* sqrt(r2);

        Ix = (quad_w.' * (dx ./ r3)) * J;
        Iy = (quad_w.' * (dy ./ r3)) * J;
        Iz = (quad_w.' * (dz ./ r3)) * J;

        Rx(m,k) = Ix;
        Ry(m,k) = Iy;
        Rz(m,k) = Iz;
    end
end

end








%% FUNCITON: compute_r_matrices_quadrature_3d
function [Rx,Ry,Rz] = compute_r_matrices_quadrature_3d(fmdl,quad_mode)

nodes    = fmdl.nodes;      % N×3
elements = fmdl.elems;      % Ntets×4
sensors  = fmdl.sensors;
n_elem    = size(elements,1);
n_sensors = length(sensors);

Rx = zeros(n_sensors,n_elem);
Ry = zeros(n_sensors,n_elem);
Rz = zeros(n_sensors,n_elem);

persistent quad_l quad_w nq;

[quad_l,quad_w,nq] = get_quadrature_3d(quad_mode);

% Precompute physical quadrature points
elem_J = zeros(1,n_elem);
elem_p = zeros(nq,3,n_elem);

for k = 1:n_elem
    v = nodes(elements(k,:),:);

    A  = v(1,:);
    BA = (v(2,:) - A).';
    CA = (v(3,:) - A).';
    DA = (v(4,:) - A).';

    elem_J(k) = abs(det([BA CA DA])) / 6;

    % Vectorized barycentric mapping
    l2 = quad_l(:,1);
    l3 = quad_l(:,2);
    l4 = quad_l(:,3);
    l1 = 1 - l2 - l3 - l4;

    % [nqx3xn_elem] quadrature point global coordinates
    elem_p(:,:,k) = ...
        l1*A + l2*v(2,:) + l3*v(3,:) + l4*v(4,:);
end

parfor m = 1:n_sensors
    c   = sensors(m).position(:).';

    % [nq × n_elem] x-y-z coordinates of vector from sensor to quadrature point
    % (numerator)
    dx = c(1) - squeeze(elem_p(:,1,:));
    dy = c(2) - squeeze(elem_p(:,2,:));
    dz = c(3) - squeeze(elem_p(:,3,:));

    % Edge case for when length(quad_w) == 1
    if size(dx,1)>size(dx,2)
        dx = dx';
        dy = dy';
        dz = dz';
    end

    % [nq × n_elem] norm of that vector (denominator)
    r2 = dx.^2 + dy.^2 + dz.^2;
    r3 = r2 .* sqrt(r2);

    % dx ./ r3 is the x-component of the integrand evaluated at all
    % quadrature points at all elements

    % [Ix,Iy,Iz] is the evaluated integral, Ix = \int_{\Omega_k}
    % (r_m-r).e_x/||(r_m-r)||^3
    
    Ix = (quad_w.' * (dx ./ r3)) .* elem_J;
    Iy = (quad_w.' * (dy ./ r3)) .* elem_J;
    Iz = (quad_w.' * (dz ./ r3)) .* elem_J;

    Rx(m,:) = Ix;
    Ry(m,:) = Iy;
    Rz(m,:) = Iz;
end

end

%% FUNCTION: compute_r_matrices_quadrature_2d
function [Rx,Ry,Rz] = compute_r_matrices_quadrature_2d(fmdl,quad_mode)

nodes    = fmdl.nodes;
elements = fmdl.elems;
sensors  = fmdl.sensors;

n_elem    = size(elements,1);
n_sensors = length(sensors);

Rx = zeros(n_sensors,n_elem);
Ry = zeros(n_sensors,n_elem);
Rz = zeros(n_sensors,n_elem);

%% 1. Precompute geometric data and quadrature points
persistent quad_l quad_w nq;
[quad_l,quad_w,nq] = get_quadrature_2d(quad_mode);

% Precompute physical quadrature points
elem_J = zeros(1,n_elem);
elem_p = zeros(nq,2,n_elem);

for k = 1:n_elem
    v = nodes(elements(k,:),:);    % A,B,C
    A = v(1,:); B = v(2,:); C = v(3,:);

    J = 0.5*abs(det([B-A; C-A]));
    elem_J(k) = J;

    % Map quadrature points
    P = zeros(size(quad_w,1),2);
    for q = 1:size(quad_w,1)
        xi  = quad_l(q,1);
        eta = quad_l(q,2);
        P(q,:) = A + xi*(B-A) + eta*(C-A);
    end
    elem_p(:,:,k) = P;
end

%% 2. Loop over sensors
parfor m = 1:n_sensors
    c = sensors(m).position;       % sensor center

    % P = elem_p{k};    % 3×2 array of quadrature points

    % [nq × n_elem] x-y-z coordinates of vector from sensor to quadrature point
    % (numerator)
    dx = c(1) - squeeze(elem_p(:,1,:));
    dy = c(2) - squeeze(elem_p(:,2,:));
    dz = c(3)*ones(nq,n_elem); %sensors and quadrature points are both assumed to be at z = 0
    
    % Edge case for when length(quad_w) == 1
    if size(dx,1)>size(dx,2)
        dx = dx';
        dy = dy';
    end

    % [nq × n_elem] norm of that vector (denominator)
    r2 = dx.^2 + dy.^2 + dz.^2;
    r3 = r2 .* sqrt(r2);

    % [Ix,Iy,Iz] is the evaluated integral, Ix = \int_{\Omega_k}
    % (r_m-r).e_x/||(r_m-r)||^3
    Ix = (quad_w.' * (dx ./ r3)) .* elem_J;
    Iy = (quad_w.' * (dy ./ r3)) .* elem_J;
    Iz = (quad_w.' * (dz ./ r3)) .* elem_J;

    % I = [Ix Iy Iz];

    Rx(m,:) = Ix;
    Ry(m,:) = Iy;
    Rz(m,:) = Iz;
end

end






%% QUADRATURE RULES (3D):
function [quad_l,quad_w,nq] = get_newton_cotes_open_6()
    quad_l = [ ...
    0.1000000000000000  0.1000000000000000  0.1000000000000000;
    0.1000000000000000  0.1000000000000000  0.7000000000000000;
    0.1000000000000000  0.7000000000000000  0.1000000000000000;
    0.7000000000000000  0.1000000000000000  0.1000000000000000;
    0.1000000000000000  0.1000000000000000  0.6000000000000000;
    0.1000000000000000  0.1000000000000000  0.2000000000000000;
    0.1000000000000000  0.6000000000000000  0.1000000000000000;
    0.1000000000000000  0.6000000000000000  0.2000000000000000;
    0.1000000000000000  0.2000000000000000  0.1000000000000000;
    0.1000000000000000  0.2000000000000000  0.6000000000000000;
    0.6000000000000000  0.1000000000000000  0.1000000000000000;
    0.6000000000000000  0.1000000000000000  0.2000000000000000;
    0.6000000000000000  0.2000000000000000  0.1000000000000000;
    0.2000000000000000  0.1000000000000000  0.1000000000000000;
    0.2000000000000000  0.1000000000000000  0.6000000000000000;
    0.2000000000000000  0.6000000000000000  0.1000000000000000;
    0.1000000000000000  0.1000000000000000  0.5000000000000000;
    0.1000000000000000  0.1000000000000000  0.3000000000000000;
    0.1000000000000000  0.5000000000000000  0.1000000000000000;
    0.1000000000000000  0.5000000000000000  0.3000000000000000;
    0.1000000000000000  0.3000000000000000  0.1000000000000000;
    0.1000000000000000  0.3000000000000000  0.5000000000000000;
    0.5000000000000000  0.1000000000000000  0.1000000000000000;
    0.5000000000000000  0.1000000000000000  0.3000000000000000;
    0.5000000000000000  0.3000000000000000  0.1000000000000000;
    0.3000000000000000  0.1000000000000000  0.1000000000000000;
    0.3000000000000000  0.1000000000000000  0.5000000000000000;
    0.3000000000000000  0.5000000000000000  0.1000000000000000;
    0.2000000000000000  0.2000000000000000  0.1000000000000000;
    0.2000000000000000  0.2000000000000000  0.5000000000000000;
    0.2000000000000000  0.1000000000000000  0.2000000000000000;
    0.2000000000000000  0.1000000000000000  0.5000000000000000;
    0.2000000000000000  0.5000000000000000  0.2000000000000000;
    0.2000000000000000  0.5000000000000000  0.1000000000000000;
    0.1000000000000000  0.2000000000000000  0.2000000000000000;
    0.1000000000000000  0.2000000000000000  0.5000000000000000;
    0.1000000000000000  0.5000000000000000  0.2000000000000000;
    0.5000000000000000  0.2000000000000000  0.2000000000000000;
    0.5000000000000000  0.2000000000000000  0.1000000000000000;
    0.5000000000000000  0.1000000000000000  0.2000000000000000;
    0.4000000000000000  0.4000000000000000  0.1000000000000000;
    0.4000000000000000  0.1000000000000000  0.4000000000000000;
    0.4000000000000000  0.1000000000000000  0.1000000000000000;
    0.1000000000000000  0.4000000000000000  0.4000000000000000;
    0.1000000000000000  0.4000000000000000  0.1000000000000000;
    0.1000000000000000  0.1000000000000000  0.4000000000000000;
    0.4000000000000000  0.3000000000000000  0.2000000000000000;
    0.4000000000000000  0.3000000000000000  0.1000000000000000;
    0.4000000000000000  0.2000000000000000  0.3000000000000000;
    0.4000000000000000  0.2000000000000000  0.1000000000000000;
    0.4000000000000000  0.1000000000000000  0.3000000000000000;
    0.4000000000000000  0.1000000000000000  0.2000000000000000;
    0.3000000000000000  0.4000000000000000  0.2000000000000000;
    0.3000000000000000  0.4000000000000000  0.1000000000000000;
    0.3000000000000000  0.2000000000000000  0.4000000000000000;
    0.3000000000000000  0.2000000000000000  0.1000000000000000;
    0.3000000000000000  0.1000000000000000  0.4000000000000000;
    0.3000000000000000  0.1000000000000000  0.2000000000000000;
    0.2000000000000000  0.4000000000000000  0.3000000000000000;
    0.2000000000000000  0.4000000000000000  0.1000000000000000;
    0.2000000000000000  0.3000000000000000  0.4000000000000000;
    0.2000000000000000  0.3000000000000000  0.1000000000000000;
    0.2000000000000000  0.1000000000000000  0.4000000000000000;
    0.2000000000000000  0.1000000000000000  0.3000000000000000;
    0.1000000000000000  0.4000000000000000  0.3000000000000000;
    0.1000000000000000  0.4000000000000000  0.2000000000000000;
    0.1000000000000000  0.3000000000000000  0.4000000000000000;
    0.1000000000000000  0.3000000000000000  0.2000000000000000;
    0.1000000000000000  0.2000000000000000  0.4000000000000000;
    0.1000000000000000  0.2000000000000000  0.3000000000000000;
    0.2000000000000000  0.2000000000000000  0.2000000000000000;
    0.2000000000000000  0.2000000000000000  0.4000000000000000;
    0.2000000000000000  0.4000000000000000  0.2000000000000000;
    0.4000000000000000  0.2000000000000000  0.2000000000000000;
    0.3000000000000000  0.3000000000000000  0.3000000000000000;
    0.3000000000000000  0.3000000000000000  0.1000000000000000;
    0.3000000000000000  0.1000000000000000  0.3000000000000000;
    0.1000000000000000  0.3000000000000000  0.3000000000000000;
    0.3000000000000000  0.3000000000000000  0.2000000000000000;
    0.3000000000000000  0.2000000000000000  0.3000000000000000;
    0.3000000000000000  0.2000000000000000  0.2000000000000000;
    0.2000000000000000  0.3000000000000000  0.3000000000000000;
    0.2000000000000000  0.3000000000000000  0.2000000000000000;
    0.2000000000000000  0.2000000000000000  0.3000000000000000;
    ] ;

quad_w = [
    0.2843915343915344
    0.2843915343915344
    0.2843915343915344
    0.2843915343915344
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    -0.3882275132275133
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.8776455026455027
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    0.1236772486772487
    -0.8584656084656085
    -0.8584656084656085
    -0.8584656084656085
    -0.8584656084656085
    -0.8584656084656085
    -0.8584656084656085
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    -0.2632275132275133
    0.0145502645502645
    0.0145502645502645
    0.0145502645502645
    0.0145502645502645
    1.0165343915343916
    1.0165343915343916
    1.0165343915343916
    1.0165343915343916
    -0.0251322751322751
    -0.0251322751322751
    -0.0251322751322751
    -0.0251322751322751
    -0.0251322751322751
    -0.0251322751322751];

nq = size(quad_l,1);
end

function [quad_l,quad_w,nq] = get_newton_cotes_open_2()
quad_l = [ ...
    0.1666666666666667  0.1666666666666667  0.1666666666666667;
    0.1666666666666667  0.1666666666666667  0.5000000000000000;
    0.1666666666666667  0.5000000000000000  0.1666666666666667;
    0.5000000000000000  0.1666666666666667  0.1666666666666667;
    0.3333333333333333  0.3333333333333333  0.1666666666666667;
    0.3333333333333333  0.1666666666666667  0.3333333333333333;
    0.3333333333333333  0.1666666666666667  0.1666666666666667;
    0.1666666666666667  0.3333333333333333  0.3333333333333333;
    0.1666666666666667  0.3333333333333333  0.1666666666666667;
    0.1666666666666667  0.1666666666666667  0.3333333333333333;
    ] ;

quad_w = [
  0.5500000000000000;
  0.5500000000000000;
  0.5500000000000000;
  0.5500000000000000;
 -0.2000000000000000;
 -0.2000000000000000;
 -0.2000000000000000;
 -0.2000000000000000;
 -0.2000000000000000;
 -0.2000000000000000];

nq = size(quad_l,1);
end

function [quad_l,quad_w,nq] = get_newton_cotes_open_0()
quad_l = [ ...
      0.2500000000000000  0.2500000000000000  0.2500000000000000
      ] ;

quad_w = 1.0000000000000000;

nq = size(quad_l,1);
end

%% QUADRATURE RULES (2D):
function [quad_l,quad_w,nq] = get_toms_37()
quad_l = [
    0.333333333333333333333333333333  0.333333333333333333333333333333;
    0.950275662924105565450352089520  0.024862168537947217274823955239;
    0.024862168537947217274823955239  0.950275662924105565450352089520;
    0.024862168537947217274823955239  0.024862168537947217274823955239;
    0.171614914923835347556304795551  0.414192542538082326221847602214;
    0.414192542538082326221847602214  0.171614914923835347556304795551;
    0.414192542538082326221847602214  0.414192542538082326221847602214;
    0.539412243677190440263092985511  0.230293878161404779868453507244;
    0.230293878161404779868453507244  0.539412243677190440263092985511;
    0.230293878161404779868453507244  0.230293878161404779868453507244;
    0.772160036676532561750285570113  0.113919981661733719124857214943;
    0.113919981661733719124857214943  0.772160036676532561750285570113;
    0.113919981661733719124857214943  0.113919981661733719124857214943;
    0.009085399949835353883572964740  0.495457300025082323058213517632;
    0.495457300025082323058213517632  0.009085399949835353883572964740;
    0.495457300025082323058213517632  0.495457300025082323058213517632;
    0.062277290305886993497083640527  0.468861354847056503251458179727;
    0.468861354847056503251458179727  0.062277290305886993497083640527;
    0.468861354847056503251458179727  0.468861354847056503251458179727;
    0.022076289653624405142446876931  0.851306504174348550389457672223;
    0.022076289653624405142446876931  0.126617206172027096933163647918;
    0.851306504174348550389457672223  0.022076289653624405142446876931;
    0.851306504174348550389457672223  0.126617206172027096933163647918;
    0.126617206172027096933163647918  0.022076289653624405142446876931;
    0.126617206172027096933163647918  0.851306504174348550389457672223;
    0.018620522802520968955913511549  0.689441970728591295496647976487;
    0.018620522802520968955913511549  0.291937506468887771754472382212;
    0.689441970728591295496647976487  0.018620522802520968955913511549;
    0.689441970728591295496647976487  0.291937506468887771754472382212;
    0.291937506468887771754472382212  0.018620522802520968955913511549;
    0.291937506468887771754472382212  0.689441970728591295496647976487;
    0.096506481292159228736516560903  0.635867859433872768286976979827;
    0.096506481292159228736516560903  0.267625659273967961282458816185;
    0.635867859433872768286976979827  0.096506481292159228736516560903;
    0.635867859433872768286976979827  0.267625659273967961282458816185;
    0.267625659273967961282458816185  0.096506481292159228736516560903;
    0.267625659273967961282458816185  0.635867859433872768286976979827];

quad_w  = [
    0.051739766065744133555179145422
    0.008007799555564801597804123460
    0.008007799555564801597804123460
    0.008007799555564801597804123460
    0.046868898981821644823226732071
    0.046868898981821644823226732071
    0.046868898981821644823226732071
    0.046590940183976487960361770070
    0.046590940183976487960361770070
    0.046590940183976487960361770070
    0.031016943313796381407646220131
    0.031016943313796381407646220131
    0.031016943313796381407646220131
    0.010791612736631273623178240136
    0.010791612736631273623178240136
    0.010791612736631273623178240136
    0.032195534242431618819414482205
    0.032195534242431618819414482205
    0.032195534242431618819414482205
    0.015445834210701583817692900053
    0.015445834210701583817692900053
    0.015445834210701583817692900053
    0.015445834210701583817692900053
    0.015445834210701583817692900053
    0.015445834210701583817692900053
    0.017822989923178661888748319485
    0.017822989923178661888748319485
    0.017822989923178661888748319485
    0.017822989923178661888748319485
    0.017822989923178661888748319485
    0.017822989923178661888748319485
    0.037038683681384627918546472190
    0.037038683681384627918546472190
    0.037038683681384627918546472190
    0.037038683681384627918546472190
    0.037038683681384627918546472190
    0.037038683681384627918546472190];  % all weights = area/6 in reference triangle

nq = size(quad_l,1);
end

function [quad_l,quad_w,nq] = get_centroid_quadrature()
quad_l = [0.33333333333333333333  0.33333333333333333333];

quad_w  = 1.00000000000000000000;  % all weights = area/6 in reference triangle

nq = size(quad_l,1);
end