function img= inv_solve_diff_GN_one_step_mdeit( inv_model, data1, data2)
% INV_SOLVE_DIFF_GN_ONE_STEP_MDEIT inverse solver using approach of Adler&Guardo 1996
% img= inv_solve_diff_GN_one_step( inv_model, data1, data2)
% img        => output image (or vector of images)
% inv_model  => inverse model struct
% data1      => differential data at earlier time
% data2      => differential data at later time
%
% both data1 and data2 may be matrices (MxT) each of
%  M measurements at T times
% if either data1 or data2 is a vector, then it is expanded
%  to be the same size matrix
%
% By default, the correct scaling of the solution that best fits the data
% is not calculated, possibly resulting in high solution errors reported by
% inv_solve. 
% To calculate the correct scaling, specify
%     inv_model.inv_solve_diff_GN_one_step.calc_step_size = 1;
% To provide a pre-calculated scaling, specify
%     inv_model.inv_solve_diff_GN_one_step.calc_step_size = 0;
%     inv_model.inv_solve_diff_GN_one_step.step_size = 0.8;
% The search for correct step_size is performed using FMINBND. The default
% search interval is [1e-5 1e1]. You can modify it by specifying:
%     inv_model.inv_solve_diff_GN_one_step.bounds = [10 200];
% Additional options for FMINBD can be passed as:
%     inv_model.inv_solve_diff_GN_one_step.fminbnd.MaxIter = 10;
%
% The optimal step_size is returned in img.info.step_size.
%
% If inv_model.inv_solve_core.do_pcg = true, the normal equations are
% solved with PCG using a MATRIX-FREE Jacobian operator
% (calc_jacobian_operator_mdeit): J and J'*J are never assembled, only
% J*v / J'*w matvecs are used inside the PCG iteration.
%
% See also INV_SOLVE, CALC_SOLUTION_ERROR, FMINBND, CALC_JACOBIAN_OPERATOR_MDEIT

% (C) 2005-2013 Andy Adler and Bartlomiej Grychtol. 
% License: GPL version 2 or version 3
% $Id: inv_solve_diff_GN_one_step.m 5611 2017-06-27 12:02:50Z htregidgo $

% TODO:
% Test whether Wiener filter form or Tikhonov form are faster
%  Tikhonov: RM= (J'*W*J +  hp^2*RtR)\J'*W;
%  Wiener:   P= inv(RtR); V = inv(W); RM = P*J'/(J*P*J' + hp^2*V)

% dv = calc_difference_data( data1, data2, inv_model.fwd_model);

dm = calc_difference_data_mdeit(data1, data2, inv_model.fwd_model);
inv_model.hyperparameter.data = dm;

sol = solve_normal_equations(inv_model, dm);

img = data_mapper(calc_jacobian_bkgnd( inv_model ));
img.name= 'solved by inv_solve_diff_GN_one_step';
img.elem_data = sol;
img.fwd_model= inv_model.fwd_model;
img = data_mapper(img,1);

% img = scale_to_fit_data(img, inv_model, data1, data2);
end

function RM = get_RM( inv_model )
img_bkgnd= calc_jacobian_bkgnd( inv_model );
J = calc_jacobian_mdeit( img_bkgnd);

RtR = calc_RtR_prior( inv_model );

% W   = calc_meas_icov( inv_model );
hp  = calc_hyperparameter_mdeit( inv_model );

% left_divide now has handling to force symmetric methods when matrices
% are symmetric up to floating point error.

% RM  = left_divide((J'*W*J +  hp^2*RtR),J'*W);
RM  = left_divide((J'*J +  hp^2*RtR),J');
end

function sol = solve_normal_equations(inv_model, dm)
do_pcg = false;
tol = 1e-4;
maxit = size(dm, 1);
jacobian_progress = false;   % live per-block counter inside J*v / J'*w matvecs

if isfield(inv_model, 'inv_solve_core')
   if isfield(inv_model.inv_solve_core, 'do_pcg')
      do_pcg = inv_model.inv_solve_core.do_pcg;
   end
   if isfield(inv_model.inv_solve_core, 'tol')
      tol = inv_model.inv_solve_core.tol;
   end
   if isfield(inv_model.inv_solve_core, 'maxit')
      maxit = inv_model.inv_solve_core.maxit;
   end
   if isfield(inv_model.inv_solve_core, 'jacobian_progress')
      jacobian_progress = inv_model.inv_solve_core.jacobian_progress;
   end
end

if do_pcg
   % Emit a warning that pcg is being used
   eidors_msg('inv_solve_diff_GN_one_step: Using PCG solver (matrix-free Jacobian)',2);

   img_bkgnd = calc_jacobian_bkgnd(inv_model);

   % Matrix-free operator: builds Gamma/forward/adjoint solves ONCE,
   % then hands back cheap J*v / J'*w closures. Neither J nor J'*J
   % is ever assembled.
   Jop = calc_jacobian_operator_mdeit(img_bkgnd, jacobian_progress);

   RtR = calc_RtR_prior(inv_model);
   hp  = calc_hyperparameter_mdeit(inv_model);

   rhs = Jop.mtimes_transp(dm);
   sol = solve_normal_equations_pcg(Jop, RtR, hp, rhs, tol, maxit);
else
   % Emit a warning that left_divide is being used
   eidors_msg('inv_solve_diff_GN_one_step: Using left_divide solver',2);
   sol = eidors_cache(@get_RM, inv_model,'inv_solve_diff_GN_one_step' ) * dm;
end
end

function sol = solve_normal_equations_pcg(Jop, RtR, hp, rhs, tol, maxit)
n_unknowns = size(rhs, 1);
n_rhs = size(rhs, 2);
sol = zeros(n_unknowns, n_rhs);

% Matrix-free application of (J'*J + hp^2*RtR)*x.
% J'*J is never assembled: each call does Jop.mtimes (a forward matvec)
% followed by Jop.mtimes_transp (an adjoint matvec), both of which
% themselves avoid forming J.
apply_normal_eq = @(x) Jop.mtimes_transp(Jop.mtimes(x)) + hp^2 * (RtR * x);

for rhs_idx = 1:n_rhs
   [sol(:, rhs_idx), flag, relres, iter] = pcg( ...
      apply_normal_eq, rhs(:, rhs_idx), tol, maxit, [], [], zeros(n_unknowns, 1));

   switch flag
      case 0
         % PCG converged successfully
      case 1
         % PCG did not converge within the maximum number of iterations
         warning('inv_solve_diff_GN_one_step_mdeit:pcg exceeded maximum iterations without converging (relres=%g, iter=%d)', relres, iter);
      case 2
         % The preconditioner matrix M or M = M1*M2 is ill conditioned.
         warning('inv_solve_diff_GN_one_step_mdeit:pcg preconditioner is ill conditioned (relres=%g, iter=%d)', relres, iter);
      case 3
         % PCG stagnated
         warning('inv_solve_diff_GN_one_step_mdeit:pcg stagnated (relres=%g, iter=%d)', relres, iter);
      case 4
         % One of the scalar quantities calculated during PCG became too small or too large to continue computing.
         error('inv_solve_diff_GN_one_step_mdeit:pcg scalar quantity too small or too large (relres=%g, iter=%d)',relres, iter);
   end
end
end