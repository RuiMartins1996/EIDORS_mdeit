function fmdl = compute_geometry_matrices(fmdl,mu0,quad_mode)

ndims = size(fmdl.nodes,2);

if nargin <3
    if ndims == 3
        quad_mode = 'nco6';
    elseif ndims ==2
        quad_mode = 'toms37';
    else
        error('Unexpected');
    end
end

% Compute and assign G and R ( calls the correct function for the 2d and 3d
% case)
[Gx,Gy,Gz,V,elementCentroids] = compute_gradient_matrix(fmdl);

[Rx,Ry,Rz] = compute_r_matrices(fmdl,quad_mode);

fmdl.G = struct('Gx',Gx,'Gy',Gy,'Gz',Gz);
fmdl.R = struct('Rx',Rx,'Ry',Ry,'Rz',Rz);
fmdl.elem_volume = V;
fmdl.elem_centroids = elementCentroids;

% Assign mu0 (must go above: needed by computeGammaMatrices)
fmdl.mu0 = mu0;
end

