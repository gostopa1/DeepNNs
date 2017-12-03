function [act,der]=squarerootact(X,W,B)
preact=X*W+repmat(B,1,size(X,1))';
act=preact.^(1/2);
der=(1/2)*preact.^(-1/2);

end