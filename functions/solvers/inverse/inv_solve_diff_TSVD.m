function img = inv_solve_diff_TSVD(inv_model, data1, data2)
% INV_SOLVE_DIFF_TSVD inverse solver using truncated singular value decomposition
%   img = inv_solve_diff_TSVD(inv_model, data1, data2)
%
%   img        => output image (or vector of images)
%   inv_model  => inverse model struct
%   data1      => differential data at earlier time
%   data2      => differential data at later time
%
% Both data1 and data2 may be matrices (MxT) each of M measurements at T
% times. If either data1 or data2 is a vector, then it is expanded to match
% the other input.

% Hyperparameter for TSVD is the number of singular values to keep
% Check the hyperparameter value
if ~isfield(inv_model.hyperparameter, 'value') || isempty(inv_model.hyperparameter.value)
    error('inv_solve_diff_TSVD: inv_model.hyperparameter.value must be set to the number of singular values to keep');
end

dm = calc_difference_data_mdeit(data1, data2, inv_model.fwd_model);
inv_model.hyperparameter.data = dm;

img = data_mapper(calc_jacobian_bkgnd(inv_model));
img.name = 'solved by inv_solve_diff_TSVD';
img.elem_data = solve_tsvd(inv_model, dm);
img.fwd_model = inv_model.fwd_model;
img = data_mapper(img, 1);

end

function sol = solve_tsvd(inv_model, dm)
	RM = eidors_cache(@get_RM, {inv_model}, 'inv_solve_diff_TSVD');
	sol = RM * dm;
end

function RM = get_RM(inv_model)

    img_bkgnd = calc_jacobian_bkgnd(inv_model);
    J = calc_jacobian_mdeit(img_bkgnd);
    

    k = inv_model.hyperparameter.value;

    [U,S,V] = svd(J, 'econ');

    Uk = U(:,1:k);  
    Vk = V(:,1:k);
    Sk = S(1:k,1:k);

    RM = Vk * diag(1./diag(Sk)) * Uk';
end

