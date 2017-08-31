function [act,der]=tanhact(X,W,B)
act=tanh(X*W+repmat(B,1,size(X,1))');

der=1-(act.^2);



end