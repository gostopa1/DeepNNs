function [error,dedout]=cross_entropy_cost(model)
%implementation based on this: https://sgugger.github.io/a-simple-neural-net-in-numpy.html

eps=1e-9;
temp_out = model.layers(end).out;
temp_out(temp_out(:)<eps)=eps; % Remove too small values to address numerical issues in log and division

temp_error = model.target.*log(temp_out);
%sum(temp_error)
error = -sum(sum(model.target.*log(temp_out)))/model.batchsize;

error = error + model.l2/(2*model.batchsize)*sum(model.allweights(:).^2)+model.l1/(2*model.batchsize)*sum(abs(model.allweights(:)));

dedout=model.target.*(-1./temp_out);
end