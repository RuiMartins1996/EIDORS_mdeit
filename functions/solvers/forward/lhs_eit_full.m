function out = lhs_eit_full(img)
% Route through calc_system_mat (not system_mat_1st_order directly) so this
% shares the eidors_cache entry with fwd_solve_1st_order_mdeit instead of
% reassembling E. mdeit_lean_img keeps the cache key cheap and must match the
% key used there, or the two callers will not share an entry.

if isfield(img.fwd_model,'system_mat')
   s_mat = calc_system_mat( mdeit_lean_img(img) );
else
   % calc_system_mat requires fwd_model.system_mat to be set. Callers that
   % build a bare fwd_model without it fall back to the uncached assembly.
   s_mat = system_mat_1st_order(img);
end

out = s_mat.E;
end
