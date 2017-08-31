function [act,der]=logsi(X,W,B)
act=1./(1+exp(-(X*W+repmat(B,1,size(X,1))')));
der=act.*(1-act);

end