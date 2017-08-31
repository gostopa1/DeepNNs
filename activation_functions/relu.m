function [act,der]=relu(X,W,B)
act=max(X*W+repmat(B,1,size(X,1))',0);
der=act;
der(act>0)=1;
end