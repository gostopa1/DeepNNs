function [act,der]=quadraticact(X,W,B)
preact=X*W+repmat(B,1,size(X,1))';
act=preact.^2;

der=2.*preact;
end