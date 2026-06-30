function [G1,G2,G3,element_volume,element_centroids] = compute_gradient_matrix(fmdl)

if size(fmdl.nodes,2) == 2
    % [G1, G2, element_volume, element_centroids]  = compute_gradient_matrix_2d(fmdl);
    [G1, G2, element_volume, element_centroids] = compute_gradient_matrix_2d_vec(fmdl);
    
    G3 = sparse(size(G1,1),size(G1,2));
elseif size(fmdl.nodes,2) == 3
    % [G1,G2,G3,element_volume,element_centroids]  = compute_gradient_matrix_3d(fmdl);
    [G1,G2,G3,element_volume,element_centroids] = compute_gradient_matrix_3d_vec(fmdl);
else
    error('Unexpected dimension for fmdl')
end

end


%% Function
function [G1,G2,G3,element_volume,element_centroids] = compute_gradient_matrix_3d(fmdl)

n_elem = size(fmdl.elems,1);
n_nodes = size(fmdl.nodes,1);

G1 = sparse(n_elem,n_nodes);
G2 = sparse(n_elem,n_nodes);
G3 = sparse(n_elem,n_nodes);

element_volume = zeros(n_elem,1);
element_centroids = zeros(n_elem,3);

for k = 1:n_elem
    nodeIds = fmdl.elems(k,:);
    nodes = zeros(3,length(nodeIds));
    for n = 1:length(nodeIds)
        nodes(:,n) = fmdl.nodes(nodeIds(n),:);
    end

    element_centroids(k,:) = sum(nodes,2)/length(nodeIds);

    B = [nodes' ones(4,1)];
    grad1 = B\[1;0;0;0];grad2 = B\[0;1;0;0];grad3 = B\[0;0;1;0];grad4= B\[0;0;0;1];
    grad1 = grad1(1:3);grad2 = grad2(1:3);grad3 = grad3(1:3);grad4 = grad4(1:3);

    element_volume(k) = 1/6*abs(det(B));

    grads = [grad1,grad2,grad3,grad4];

    G1(k,nodeIds) = G1(k,nodeIds) + grads(1,:);
    G2(k,nodeIds) = G2(k,nodeIds) + grads(2,:);
    G3(k,nodeIds) = G3(k,nodeIds) + grads(3,:);
end

end

%% CHAT GPT otpimized version
function [G1, G2, G3, element_volume, element_centroids] = compute_gradient_matrix_3d_vec(fmdl)
% Vectorized exact reproduction of the original per-element B \ e approach.
% Requires pageinv (R2021b+) — which you indicated you have.

elems = fmdl.elems;    % n_elem x 4
nodes = fmdl.nodes;    % n_nodes x 3

n_elem  = size(elems,1);
n_nodes = size(nodes,1);

% Vertex coordinates for all elements: X is 3 x 4 x n_elem
X = permute(reshape(nodes(elems',:).', 3, 4, n_elem), [1 2 3]);

% Centroids
element_centroids = squeeze(mean(X,2)).';  % n_elem x 3

% Build B for each element: B_k is 4x4 such that rows are [x y z 1] for each vertex
% We want B(:,:,k) to be 4x4xn_elem
% Create the first three columns by transposing X appropriately:
% For element k, rows should be: [X(:,1,k)' 1; X(:,2,k)' 1; X(:,3,k)' 1; X(:,4,k)' 1]
B = zeros(4,4,n_elem);
B(1:4,1:3,:) = permute(X, [2 1 3]);   % becomes 4 x 3 x n_elem, rows are vertices
B(1:4,4,:) = 1;                       % last column ones

% Invert all B's at once
invB = pageinv(B);   % 4 x 4 x n_elem

% Gradients: for element k, gradient of basis j is invB(1:3, j, k)
% So build grads as 3 x 4 x n_elem
grads = invB(1:3, :, :);   % 3 x 4 x n_elem

% Volume: your loop used element_volume = 1/6*abs(det(B))
% For the 4x4 B defined above, det(B) = det([x1 x2 x3 x4;1...]?) 
% Equivalent and robust approach: use triple product via X (faster, stable)
v1 = squeeze(X(:,2,:) - X(:,1,:));   % 3 x n_elem
v2 = squeeze(X(:,3,:) - X(:,1,:));
v3 = squeeze(X(:,4,:) - X(:,1,:));
cross23 = cross(v2, v3, 1);          % 3 x n_elem
triple  = dot(v1, cross23, 1);       % 1 x n_elem
element_volume = abs(triple(:)) / 6; % n_elem x 1

% Assemble sparse matrices G1,G2,G3
rows = repelem((1:n_elem)', 4);      % 4*n_elem x 1
temp = elems';
cols = temp(:);                     % 4*n_elem x 1

G1vals = reshape(grads(1,:,:), 4*n_elem, 1);
G2vals = reshape(grads(2,:,:), 4*n_elem, 1);
G3vals = reshape(grads(3,:,:), 4*n_elem, 1);

G1 = sparse(rows, cols, G1vals, n_elem, n_nodes);
G2 = sparse(rows, cols, G2vals, n_elem, n_nodes);
G3 = sparse(rows, cols, G3vals, n_elem, n_nodes);

end

%% Function
function [Gx, Gy, element_area, element_centroids] = compute_gradient_matrix_2d(fmdl)

n_elem = size(fmdl.elems, 1);
n_nodes = size(fmdl.nodes, 1);

Gx = sparse(n_elem, n_nodes);
Gy = sparse(n_elem, n_nodes);

element_area = zeros(n_elem, 1);
element_centroids = zeros(n_elem, 2);

for k = 1:n_elem
    nodeIds = fmdl.elems(k, :);
    nodes = fmdl.nodes(nodeIds, :); % 3x2 matrix: [x1 y1; x2 y2; x3 y3]

    % Compute centroid
    element_centroids(k, :) = mean(nodes, 1);

    % Compute area (half of parallelogram area)
    x = nodes(:,1);
    y = nodes(:,2);
    A = 0.5*abs(det([ones(3,1), x, y]));
    element_area(k) = abs(A);

    % Compute gradients of basis functions
    % The local shape functions Ni = ai + bi*x + ci*y
    % We solve for [ai; bi; ci] from Ni(nodes_j) = δij
    B = [ones(3,1), x, y]; % 3x3
    grads = zeros(2,3); % rows: (dNi/dx, dNi/dy), columns: node i
    for i = 1:3
        e = zeros(3,1); e(i) = 1; 
        
        coeffs = B \ e; % [ai; bi; ci]

        grads(:, i) = coeffs(2:3); %gradient of shape function i is [bi, ci]
    end

    % Assign to sparse matrices
    Gx(k, nodeIds) = grads(1, :);
    Gy(k, nodeIds) = grads(2, :);
end

end

%% CHAT GPT optimized version
function [Gx, Gy, element_area, element_centroids] = compute_gradient_matrix_2d_vec(fmdl)
% Fully vectorized 2D gradient computation using B \ e approach

elems  = fmdl.elems;   % n_elem x 3
nodes  = fmdl.nodes;   % n_nodes x 2
n_elem = size(elems,1);
n_nodes = size(nodes,1);

% ----------------------------------------
% Extract coordinates per element
% ---------------------------------------- 
X = permute(reshape(nodes(elems',:).', 2,3, n_elem), [2 1 3]);

% ----------------------------------------
% Compute centroids
% ----------------------------------------
element_centroids = squeeze(mean(X,1))';  % n_elem x 2

% ----------------------------------------
% Construct B matrices for all elements
% ----------------------------------------
B = zeros(3,3,n_elem);
B(:,1,:) = 1;
B(:,2:3,:) = X;

% ----------------------------------------
% Compute element areas (vectorized 3x3 det)
% ----------------------------------------
a = squeeze(B(1,1,:)); b = squeeze(B(1,2,:)); c = squeeze(B(1,3,:));
d = squeeze(B(2,1,:)); e = squeeze(B(2,2,:)); f = squeeze(B(2,3,:));
g = squeeze(B(3,1,:)); h = squeeze(B(3,2,:)); i = squeeze(B(3,3,:));

element_area = 0.5 * abs(a.*(e.*i - f.*h) - b.*(d.*i - f.*g) + c.*(d.*h - e.*g));

% ----------------------------------------
% Compute gradients of shape functions (fully vectorized)
% grad_i = first 2 rows of B^-1 * e_i
% ----------------------------------------
invB = pageinv(B);          % 3 x 3 x n_elem
I = eye(3);                  % 3 x 3 identity
grads_all = pagemtimes(invB, I);  % 3 x 3 x n_elem
grads = grads_all(2:3,:,:);  % 2 x 3 x n_elem

% ----------------------------------------
% Assemble sparse matrices
% ----------------------------------------
rows = double(repelem((1:n_elem)',3));
cols = double(reshape(elems', [], 1));

Gx = sparse(rows, cols, reshape(grads(1,:,:), [], 1), n_elem, n_nodes);
Gy = sparse(rows, cols, reshape(grads(2,:,:), [], 1), n_elem, n_nodes);

end

