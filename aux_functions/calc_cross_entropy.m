function [error]=calc_cross_entropy(y_target, y_predicted)

% Restict values to range (0.001,0.999) to avoid NaNs and Infs in the
% gradient calculation.
batchsize=size(y_target,1);
error = -sum(sum(y_target.*log(y_predicted)+(1-y_target).*log(1-y_predicted),1))/batchsize;

%error = error + model.l2/(2*model.batchsize)*sum(model.allweights(:).^2)+model.l1/(2*model.batchsize)*sum(abs(model.allweights(:)));
%error = error -model.l2/(2*model.batchsize)*sum(model.allweights(:).^2)-model.l1/(2*model.batchsize)*sum(abs(model.allweights(:)));

uplim=0.99;
lowlim=0.11;

error=real(error);
end