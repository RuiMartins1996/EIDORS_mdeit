function hyperparameter = calc_hyperparameter_mdeit( inv_model )
% CALC_HYPERPARAMETER_MDEIT: calculate hyperparameter value for MDEIT
%   The hyperparameter can be either provided directly,
%     or can be based on an automatic selection approach
% 
% calc_hyperparameter_mdeit can be called as
%    hyperparameter = calc_hyperparameter_mdeit( inv_model )
% where inv_model    is an inv_model structure
%
% if inv_model.hyperparameter.func exists, it will be
%   called (must be either 'gcv' or 'lcurve'),
%   otherwise inv_model.hyperparameter.value will be returned
%
% See also CALC_HYPERPARAMETER

% (C) 2005 Andy Adler. License: GPL version 2 or version 3
% Modified for MDEIT

if isfield( inv_model.hyperparameter, 'func')
   func_name = inv_model.hyperparameter.func;
   
   % Validate that func is either 'gcv' or 'lcurve'
   if ischar(func_name)
      if ~ismember(func_name, {'gcv', 'l_curve','tsvd'})
         error(['calc_hyperparameter_mdeit: func must be either ''gcv'' or ''l_curve'' or ''tsvd'', ' ...
                'but got ''%s'''], func_name);
      end
      inv_model.hyperparameter.func = str2func(func_name);
   else
      % If it's already a function handle, check its name
      func_str = func2str(inv_model.hyperparameter.func);
      if ~ismember(func_str, {'gcv', 'l_curve','tsvd'})
         error(['calc_hyperparameter_mdeit: func must be either ''gcv'' or ''l_curve'' or ''tsvd'', ' ...
                'but got ''%s'''], func_str);
      end
   end
   
   hyperparameter = eidors_cache(...
                    inv_model.hyperparameter.func, {inv_model}, 'hyperparameter');
else
    hyperparameter = inv_model.hyperparameter.value;
end
