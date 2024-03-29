function [act,der]=softmaxact(X,W,B)

preact=X*W+repmat(B,1,size(X,1))';

% Prevent overflows by removing the largest activation value per batch element
preact=preact-max(preact,[],2); 
exppreact=exp(preact);

%Normalize to sum to 1
act=exppreact./(repmat(sum(exppreact,2),1, size(exppreact,2)));

der=act.*(1-act); % This corresponds to the diagonal of the Jacobian

end
