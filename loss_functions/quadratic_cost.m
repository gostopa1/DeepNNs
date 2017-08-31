function [error,dedout]=quadratic_cost(model)
error=sum(mean(0.5*(model.layers(end).out-model.target).^2));
dedout=(model.layers(end).out-model.target);

end