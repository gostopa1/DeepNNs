function [error,dedout]=cross_entropy_cost(model)

% Restict values to range (0.001,0.999) to avoid NaNs and Infs in the
% gradient calculation.
uplim=0.9999;
lowlim=0.0001;
model.layers(end).out(model.layers(end).out>uplim)=uplim; 
model.layers(end).out(model.layers(end).out<lowlim)=lowlim;

error=-sum(mean(model.target.*log(model.layers(end).out)+(1-model.target).*log(1-model.layers(end).out),1));
dedout=-((model.target./model.layers(end).out)-((1-model.target)./(1-model.layers(end).out)));

end