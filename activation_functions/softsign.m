function [act,der]=softsign(X,W,B)
preact=X*W+repmat(B,1,size(X,1))';

act=preact./(1+abs(preact));
der=1./(1+abs(preact)).^2;
end