function [error]=calc_cross_entropy(y_target, y_predicted)

% Restict values to range (0.001,0.999) to avoid NaNs and Infs in the
% gradient calculation.
batchsize=size(y_target,1);

eps=1e-9;
y_predicted(y_predicted(:)<eps)=eps;
error = -sum(sum(y_target.*log(y_predicted)))/batchsize;

end