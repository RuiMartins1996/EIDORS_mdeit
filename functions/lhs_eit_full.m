function out = lhs_eit_full(img)

s_mat = system_mat_1st_order(img);

out = s_mat.E;
end
