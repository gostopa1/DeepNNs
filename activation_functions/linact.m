function [act,der]=linact(X,W,B)
act=(X*W+repmat(B,1,size(X,1))');
der=ones(size(act));

end