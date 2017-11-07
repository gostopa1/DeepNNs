function [error,dedout]=cross_entropy_cost(model)

% Restict values to range (0.001,0.999) to avoid NaNs and Infs in the
% gradient calculation.

%error=-sum(mean(model.target.*log(model.layers(end).out)+(1-model.target).*log(1-model.layers(end).out),1));
error=-sum(sum(model.target.*log(model.layers(end).out)+(1-model.target).*log(1-model.layers(end).out),1))/model.batchsize;
%error=-sum(sum(model.target.*log(model.layers(end).out)));
error=error +model.l2/(2*model.batchsize)*sum(model.allweights(:).^2)+model.l1/(2*model.batchsize)*sum(abs(model.allweights(:)));
%error=error -model.l2/(2*model.batchsize)*sum(model.allweights(:).^2)-model.l1/(2*model.batchsize)*sum(abs(model.allweights(:)));

%dedout=-((model.target./model.layers(end).out)-((1-model.target)./(1-model.layers(end).out)));
dedout=(model.layers(end).out)-(model.target);



uplim=0.99;
lowlim=0.11;
model.layers(end).out(model.layers(end).out>uplim)=uplim; 
model.layers(end).out(model.layers(end).out<lowlim)=lowlim;
error=real(error);
end