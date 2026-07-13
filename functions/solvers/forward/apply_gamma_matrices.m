function B = apply_gamma_matrices(img, u)
% APPLY_GAMMA_MATRICES Compute the measurement vector B = Gamma*u without
% ever forming Gamma.
%
% Mathematically identical to compute_gamma_matrices followed by Gamma*u, but
% applies R, Sigma and G to u right-to-left, so the largest array touched is
% n_sensors x n_stim instead of n_sensors x n_nodes. Use this in the forward
% solve; the Jacobian needs Gamma explicitly (it solves A\(-Gamma')) and must
% keep calling COMPUTE_GAMMA_MATRICES.
%
% Returns B already flattened, matching the layout of the existing solver:
% [nsens*nstim x 1] for mdeit1, [3*nsens*nstim x 1] for mdeit3.

recon_mode = check_recon_mode(img);

mu_factor = img.fwd_model.mu0/(4*pi);
elem_data = img.elem_data(:);

R = img.fwd_model.R;
G = img.fwd_model.G;

% n_elem x n_stim each - sparse products, cheap
px = elem_data .* (G.Gx * u);
py = elem_data .* (G.Gy * u);
pz = elem_data .* (G.Gz * u);

% Each R is used twice; stack the two right-hand sides so R is streamed from
% memory once instead of twice. R is dense n_sensors x n_elem and this is the
% bandwidth-dominant step, so this halves the traffic.
n  = size(u,2);
Ax = R.Rx * [py, pz];   Rx_py = Ax(:,1:n);   Rx_pz = Ax(:,n+1:end);
Ay = R.Ry * [px, pz];   Ry_px = Ay(:,1:n);   Ry_pz = Ay(:,n+1:end);
Az = R.Rz * [px, py];   Rz_px = Az(:,1:n);   Rz_py = Az(:,n+1:end);

% n_sensors x n_stim
Cxu = -Rz_py + Ry_pz;
Cyu = -Rx_pz + Rz_px;
Czu = -Ry_px + Rx_py;

num_sensors = numel(img.fwd_model.sensors);

switch recon_mode
   case 'mdeit1'
      g = zeros(num_sensors, 3);
      for m = 1:num_sensors
         g(m,:) = img.fwd_model.sensors(m).axes.axis;
      end

      B = mu_factor * ( g(:,1).*Cxu + g(:,2).*Cyu + g(:,3).*Czu );
      B = B(:);
   case 'mdeit3'
      g = zeros(num_sensors,3,3);
      for m = 1:num_sensors
         g(m,:,:) = [...
             img.fwd_model.sensors(m).axes.axis1;
             img.fwd_model.sensors(m).axes.axis2;
             img.fwd_model.sensors(m).axes.axis3];
      end

      Bx = mu_factor * ( g(:,1,1).*Cxu + g(:,1,2).*Cyu + g(:,1,3).*Czu );
      By = mu_factor * ( g(:,2,1).*Cxu + g(:,2,2).*Cyu + g(:,2,3).*Czu );
      Bz = mu_factor * ( g(:,3,1).*Cxu + g(:,3,2).*Cyu + g(:,3,3).*Czu );
      B  = [Bx(:); By(:); Bz(:)];
   otherwise
      error('Unknown reconstruction mode: %s', recon_mode);
end

end


% Check which reconstruction mode is being used based on the sensor axes
% fields. Same test used by compute_gamma_matrices; factored into one local
% so it isn't duplicated within this file.
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
