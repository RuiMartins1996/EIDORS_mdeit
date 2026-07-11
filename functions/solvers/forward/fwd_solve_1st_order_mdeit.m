function data = fwd_solve_1st_order_mdeit(img)
% FWD_SOLVE_1ST_ORDER: data= fwd_solve_1st_order( img)
% First order FEM forward solver
% Input:
%    img       = image struct
%    select_sensor_axis = indices of sensor axes to use
% Output:
%    data = measurements struct
% Options: (to return internal FEM information)
%    img.fwd_solve.get_all_meas = 1 (data.volt = all FEM nodes, but not CEM)
%    img.fwd_solve.get_all_nodes= 1 (data.volt = all nodes, including CEM)
%    img.fwd_solve.get_elec_curr= 1 (data.elec_curr = current on electrodes)
%
% Model Reduction: use precomputed fields to reduce the size of
%    the forward solution. Nodes which are 1) not used in the output
%    (i.e. not electrodes) 2) all connected to the same conductivity via
%    the c2f mapping are applicable.
% see: Model Reduction for FEM Forward Solutions, Adler & Lionheart, EIT2016
%
%    img.fwd_model.model_reduction = @calc_model_reduction;
%       where the functionputs a struct with fields: main_region, regions
%       OR
%    img.fwd_model.model_reduction.main_region = vector, and 
%    img.fwd_model.model_reduction.regions = struct

% (C) 1995-2017 Andy Adler. License: GPL version 2 or version 3
% $Id: fwd_solve_1st_order.m 6308 2022-04-20 18:00:26Z aadler $

% correct input paralemeters if function was called with only img

fwd_model= img.fwd_model;

img = data_mapper(img);
if ~ismember(img.current_params, supported_params)
    error('EIDORS:PhysicsNotSupported', '%s does not support %s', ...
    'FWD_SOLVE_1ST_ORDER',img.current_params);
end
% all calcs use conductivity
img = convert_img_units(img, 'conductivity');

pp= fwd_model_parameters( fwd_model, 'skip_VOLUME' );
pp = set_gnd_node(fwd_model, pp);
s_mat= calc_system_mat( img );

if isfield(fwd_model,'model_reduction')
   [s_mat.E, main_idx, pp] = mdl_reduction(s_mat.E, ...
           img.fwd_model.model_reduction, img, pp );
else
   pp.mr_mapper = 1:size(s_mat.E,1);
end

% Normally EIT uses current stimulation. In this case there is
%  only a ground node, and this is the only dirichlet_nodes value.
%  In that case length(dirichlet_nodes) is 1 and the loop runs once
% If voltage stimulation is done, then we need to loop on the
%  matrix to calculate faster. 
[dirichlet_nodes, dirichlet_values, neumann_nodes, has_gnd_node]= ...
         find_dirichlet_nodes( fwd_model, pp );

v= full(horzcat(dirichlet_values{:})); % Pre fill in matrix
for i=1:length(dirichlet_nodes)
   idx= 1:size(s_mat.E,1);
   idx( dirichlet_nodes{i} ) = [];
   % If all dirichlet patterns are the same, then calc in one go
   if length(dirichlet_nodes) == 1; rhs = 1:size(pp.QQ,2);
   else                           ; rhs = i; end
   E   = s_mat.E(idx,idx);
   RHS = neumann_nodes{i}(idx,:) - s_mat.E(idx,:)*dirichlet_values{i};
   v(idx,rhs) = solve_reduced_system(E, RHS, fwd_model);
end

% If model has a ground node, check if current flowing in this node
if has_gnd_node
   Ignd = s_mat.E(dirichlet_nodes{1},:)*v;
   Irel = Ignd./sum(abs(pp.QQ)); % relative 
   if max(abs(Irel))>1e-6
      eidors_msg('%4.5f%% of current is flowing through ground node. Check stimulation pattern', max(abs(Irel))*100,1);
   end
end

%% Calculate the magnetic field on magnetometers 

img = compute_gamma_matrices(img);

u = v(1:size(img.fwd_model.nodes,1),:);

% Check which reconstruction mode is being used based on the sensor axes fields
if isfield(img.fwd_model.sensors(1).axes, 'axis')
    recon_mode = 'mdeit1';
elseif isfield(img.fwd_model.sensors(1).axes, 'axis1') && ...
       isfield(img.fwd_model.sensors(1).axes, 'axis2') && ...
       isfield(img.fwd_model.sensors(1).axes, 'axis3')
    recon_mode = 'mdeit3';
else
    error('Unknown sensor axes configuration. Please check the sensor axes fields.');
end

switch recon_mode
   case 'mdeit1'
      B = img.Gamma*u; % Measurement in the sensor axis direction for every sensor for every stimulation pattern
      B = B(:); % Flatten the matrix into a vector
   case 'mdeit3'
        Bx = img.Gamma1*u; % Measurement in axis1 direction for every sensor for every stimulation pattern
        By = img.Gamma2*u;
        Bz = img.Gamma3*u;
        B = [Bx(:); By(:); Bz(:)];
   otherwise
        error('Unknown reconstruction mode: %s', recon_mode);
end

% calc voltage on electrodes
% idx = find(any(pp.N2E));
% v_els= pp.N2E(:,idx) * v(idx,:);

%% Create a data structure to return

% create a data structure to return
data.type = 'data';
data.name= 'solved by fwd_solve_1st_order_mdeit';
data.time= NaN; % unknown

data.meas = B;

try; if img.fwd_solve.get_all_meas == 1
   outmap = pp.mr_mapper(1:pp.n_node);
   data.volt(outmap,:) = v(1:pp.n_node,:); % but not on CEM nodes
end; end
try; if img.fwd_solve.get_all_nodes== 1
   data.volt(pp.mr_mapper,:) = v;                % all, including CEM nodes
end; end
try; if img.fwd_solve.get_elec_curr== 1
%  data.elec_curr = pp.N2E * s_mat.E * v;
   idx = find(any(pp.N2E));
   data.elec_curr = pp.N2E(:,idx) * s_mat.E(idx,:) * v;
end; end


% has_gnd_node = flag if the model has a gnd_node => can warn if current flows
function [dirichlet_nodes, dirichlet_values, neumann_nodes, has_gnd_node]= ...
            find_dirichlet_nodes( fwd_model, pp );
   fnanQQ = isnan(pp.QQ);
   lstims = size(pp.QQ,2);
   % Can't use any(...) because if does implicit all
   if any(any(fnanQQ))
      has_gnd_node = 0; % no ground node is specified
      % Are all dirichlet_nodes the same

      % Don't use all on sparse, it will make them full
      % Check if all rows are the same
      if ~any(any(fnanQQ(:,1)*ones(1,lstims) - fnanQQ,2))
         dirichlet_nodes{1} = find(fnanQQ(:,1));
         dirichlet_values{1} = sparse(size(pp.N2E,2), size(fnanQQ,2));
         dirichlet_values{1}(fnanQQ) = pp.VV(fnanQQ);
         neumann_nodes{1} = pp.QQ;
         neumann_nodes{1}(fnanQQ) = 0;
      else % one at a time
         for i=1:size(fnanQQ,2)
            fnanQQi= fnanQQ(:,i);
            if any(fnanQQi)
               dirichlet_nodes{i} = find(fnanQQi);
               dirichlet_values{i} = sparse(size(pp.N2E,2), 1);
               dirichlet_values{i}(fnanQQi) = pp.VV(fnanQQi,i);
               neumann_nodes{i} = pp.QQ(:,i);
               neumann_nodes{i}(fnanQQi) = 0;
            elseif isfield(pp,'gnd_node')
               dirichlet_nodes{i} = pp.gnd_node;
               dirichlet_values{i} = sparse(size(pp.N2E,2), 1);
               neumann_nodes{1}   = pp.QQ(:,i);
               has_gnd_node= 1;
            else
               error('no required ground node on model');
            end 
         end
      end
   elseif isfield(pp,'gnd_node')
      dirichlet_nodes{1} = pp.gnd_node;
      dirichlet_values{1} = sparse(size(pp.N2E,2), size(fnanQQ,2));
      neumann_nodes{1}   = pp.QQ;
      has_gnd_node= 1;
   else
      error('no required ground node on model');
   end

function pp = set_gnd_node(fwd_model, pp);
   if isfield(fwd_model,'gnd_node');
      pp.gnd_node = fwd_model.gnd_node;
   else
      % try to find one in the model center
      ctr =  mean(fwd_model.nodes,1);
      d2  =  sum(bsxfun(@minus,fwd_model.nodes,ctr).^2,2);
      [~,pp.gnd_node] = min(d2);
      eidors_msg('Warning: no ground node found: choosing node %d',pp.gnd_node(1),1);
   end

% Solve the reduced linear system E*X = RHS.
% Default is EIDORS left_divide (unchanged behavior). Set
%    fwd_model.solve_mdeit.method = 'pcg' to use preconditioned conjugate
% gradient, optionally with fwd_model.solve_mdeit.tol and
% fwd_model.solve_mdeit.maxit.
function X = solve_reduced_system(E, RHS, fwd_model)
   method = 'left_divide';
   if isfield(fwd_model,'solve_mdeit') && isfield(fwd_model.solve_mdeit,'method')
      method = fwd_model.solve_mdeit.method;
   end

   switch method
      case 'left_divide'
         X = left_divide(E, RHS, fwd_model);
      case 'pcg'
         tol = 1e-8; maxit = min(size(E,1), 1000);
         if isfield(fwd_model,'solve_mdeit') && isfield(fwd_model.solve_mdeit,'tol')
            tol = fwd_model.solve_mdeit.tol;
         end
         if isfield(fwd_model,'solve_mdeit') && isfield(fwd_model.solve_mdeit,'maxit')
            maxit = fwd_model.solve_mdeit.maxit;
         end
         nrhs  = size(RHS,2);
         X     = zeros(size(RHS,1), nrhs);

         % Incomplete-Cholesky preconditioner. E (ground node removed) is
         % symmetric positive-definite, so ichol makes CG converge in far
         % fewer iterations. Without it, unpreconditioned CG stalls on large
         % (fine-mesh) systems and returns a residual larger than the small
         % B-field difference used for difference imaging -> speckle noise.
         % Built once here (same E for every RHS) and reused by all workers.
         Lic = build_ichol(E);
         % Progress counter: parfor completes out of order, so tick a
         % client-side counter as each RHS finishes via a DataQueue.
         pcg_progress('reset', nrhs);
         dq = parallel.pool.DataQueue;
         afterEach(dq, @(~) pcg_progress('tick'));
         % Right-hand sides are independent, so solve them in parallel.
         % Requires the Parallel Computing Toolbox; parfor falls back to a
         % serial loop if no pool is available.
         % NB: don't call eidors_msg inside parfor - it uses dbstack(s(2))
         % which fails on workers. error() is safe: it aborts the loop and
         % rethrows on the client.
         parfor j = 1:nrhs
            % Note: the standalone reference version passed tol*norm(RHS(:,j))
            % as the tolerance. We pass tol directly because pcg already
            % applies it relative to norm(b).
            [X(:,j), flag, relres, iters] = ...
               pcg(E, RHS(:,j), tol, maxit, Lic, Lic');

            % Fail loudly. A non-converged solve corrupts the difference
            % signal dm = datai.meas - datah.meas (a small difference of two
            % large fields), so the reconstruction would silently degrade to
            % noise. Stop here rather than return garbage.
            if flag ~= 0
                error(['fwd_solve_1st_order_mdeit: pcg column %d did not ' ...
                    'converge (flag %d, relres %.2e, %d iters, maxit %d). ' ...
                    'Increase solve_mdeit.maxit or tighten the ' ...
                    'preconditioner.'], j, flag, relres, iters, maxit);
            end

            send(dq, j);
         end
         eidors_msg(['fwd_solve_1st_order_mdeit: pcg converged for all %d ' ...
            'RHS (tol %.2e, maxit %d)'], nrhs, tol, maxit, 2);
      otherwise
         error('Unknown linear solver method: %s', method);
   end

% Client-side progress counter for the pcg solves. State is kept in
% persistent variables; call with 'reset' before the loop and 'tick'
% (via the DataQueue afterEach callback) as each RHS completes.
function pcg_progress(action, total)
   persistent count ntot
   switch action
      case 'reset'
         count = 0; ntot = total;
      case 'tick'
         count = count + 1;
         fprintf('\r  pcg: solved %d/%d RHS (%3.0f%%)', ...
                 count, ntot, 100*count/ntot);
         if count >= ntot; fprintf('\n'); end
   end

% Incomplete-Cholesky factor used as a symmetric preconditioner (M = L*L')
% for pcg. E is SPD (ground node removed), but ill-scaling can make ichol
% encounter a non-positive pivot; retry with an increasing diagonal shift
% (diagcomp) until it succeeds, then fall back to no preconditioner.
function Lic = build_ichol(E)
   % A modest drop tolerance keeps the factor sparse (fits in memory where a
   % full direct factorization does not) while still accelerating CG well.
   base_opts = struct('type','ict','droptol',1e-3,'michol','on');
   % Heuristic starting shift (Manteuffel): 0 unless the matrix needs it.
   alpha = 0;
   fprintf('fwd_solve_1st_order_mdeit: Building iChol preconditioner\n');
   for attempt = 1:8
      opts = base_opts; opts.diagcomp = alpha;
      try
         Lic = ichol(E, opts);
         return
      catch
         if alpha == 0
            % First failure: seed a shift from the diagonal dominance ratio.
            dg = full(diag(E));
            alpha = max( max(sum(abs(E),2)./max(dg,eps)) - 2, 1e-4 );
         else
            alpha = alpha * 2;   % escalate
         end
      end
   end
   % Could not build a factor: fall back to unpreconditioned CG. pcg accepts
   % [] as "no preconditioner"; the convergence check downstream still guards
   % against a bad solve.
   eidors_msg(['fwd_solve_1st_order_mdeit: ichol preconditioner failed; ' ...
      'falling back to unpreconditioned pcg'], 1);
   Lic = [];

function [E, m_idx, pp] = mdl_reduction(E, mr, img, pp);
   % if mr is a string we assume it's a function name
   if isa(mr,'function_handle') || ischar(mr)
      mr = feval(mr,img.fwd_model);
   end
   % mr is now a struct with fields: main_region, regions
   m_idx = mr.main_region;
   E = E(m_idx, m_idx);
   for i=1:length(mr.regions)
      invEi=   mr.regions(i).invE;
% FIXME:!!! data_mapper has done the c2f. But we don't want that here.
%  kludge is to reach into the fine model field. This is only ok because
%  model_reduction is only valid if one parameter describes each field
      field = mr.regions(i).field; 
      field = find(img.fwd_model.coarse2fine(:,field));
      field = field(1); % they're all the same - by def of model_reduction
      sigma = img.elem_data(field);
      E = E - sigma*invEi;
   end

   % Adjust the applied current and measurement matrices
   pp.QQ = pp.QQ(m_idx,:);
   pp.VV = pp.VV(m_idx,:);
   pp.N2E= pp.N2E(:,m_idx);
   pp.mr_mapper = cumsum(m_idx); %must be logical
   pp.gnd_node = pp.mr_mapper(pp.gnd_node);
   if pp.gnd_node==0
      error('model_reduction removes ground node');
   end
