function [data, s_mat_E] = fwd_solve_sys_mat_mdeit(img,skip_mag_field)
% FWD_SOLVE_SYS_MAT_MDEIT  Solve the mDEIT forward model and return the
% system matrix, computing system_mat_1st_order only ONCE.
%
%   [data, s_mat_E] = fwd_solve_sys_mat_mdeit(img)
%
% This is a convenience entry point for callers (e.g. the Jacobian
% solvers) that need BOTH the forward solution and the FEM system matrix.
% Doing
%     u        = fwd_solve(img);       % builds the system matrix once
%     A_matrix = lhs_eit_full(img);    % builds the system matrix AGAIN
% assembles the system matrix twice. This function assembles it once and
% hands back both results.
%
% Input:
%    img = image struct (as for fwd_solve)
% Output:
%    data    = measurements struct, identical to fwd_solve(img)
%    s_mat_E = full FEM system matrix, identical to lhs_eit_full(img)
%
% To also return the internal FEM node voltages set, as usual,
%    img.fwd_solve.get_all_meas = 1;   % data.volt = all FEM nodes
%
% See also FWD_SOLVE_1ST_ORDER_MDEIT, LHS_EIT_FULL, FWD_SOLVE.

if nargin < 2 || isempty(skip_mag_field)
    skip_mag_field = false;
end

[data, s_mat_E] = fwd_solve_1st_order_mdeit(img,skip_mag_field);

end
