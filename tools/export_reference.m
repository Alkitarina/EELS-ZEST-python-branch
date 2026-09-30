function export_reference(upstream_root, results_file)
% Export a permitted local result; private input data are not distributed.
arguments
    upstream_root (1,:) char
    results_file (1,:) char
end
project=fileparts(fileparts(mfilename('fullpath')));addpath(upstream_root);
out=fullfile(project,'reference');if ~isfolder(out),mkdir(out);end
settings=EELS_PearsonSparse_FullFit.defaults(struct('crop_range',[-250 250]));
save(fullfile(out,'defaults.mat'),'settings','-v7');
load(results_file,'results','s');
r=results{1,1};ene=r.ene_corrected;y=r.raw_data_norm;theta=r.theta;EM=r.EM_final;
obj_settings=s;obj_settings.inner_solver='lsqlin';obj_settings.tail_overshoot_constraint=false;obj_settings.lambda_pm_l1=0;
fit=EELS_PearsonSparse_FullFit.decode_stage1(theta,ene,y,EM,obj_settings);
params=EELS_PearsonSparse_FullFit.decode_theta(theta,s);
[A,layout,zlp_shapes,kernels]=EELS_PearsonSparse_FullFit.build_design_matrix(ene,EM,params,s);
weights=EELS_PearsonSparse_FullFit.build_weights(ene,y,s);
[R,r0]=EELS_PearsonSparse_FullFit.regularization_rows(EM,layout,params.T_soft,size(A,2),s,A,ene,y);
stage2=results{1,3}.stage2;expected=results{1,3};
save(fullfile(out,'pixel_reference.mat'),'r','ene','y','theta','EM','fit','A','layout','zlp_shapes','kernels','weights','R','r0','s','obj_settings','stage2','expected','-v7');
fprintf('REFERENCE_EXPORTED\n');

end
