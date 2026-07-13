function img = mdeit_lean_img(img)
% MDEIT_LEAN_IMG Strip MDEIT-only payload from img.fwd_model.
%
% EIDORS caches on a hash of the whole fwd_model (see calc_system_mat and
% fwd_model_parameters). fmdl.R holds dense n_sensors x n_elem matrices, so
% hashing the full model costs more than the FEM assembly it is meant to
% avoid. The FEM system matrix and fwd_model_parameters do not read R, G,
% sensors or mu0, so removing them changes no result and makes the cache key
% cheap. Everything the FEM needs (nodes, elems, electrode, stimulation,
% coarse2fine, background, system_mat) is kept.

heavy = {'R','G','sensors','mu0'};
img.fwd_model = rmfield(img.fwd_model, ...
    heavy(isfield(img.fwd_model, heavy)));
end
