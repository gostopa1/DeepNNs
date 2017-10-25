function [act,der]=maxpool(X,W,B)

sW=size(W);
sX=size(X);
XX=repmat(X,1,1,sW(2));
WW=permute(repmat(W,1,1,sX(1)),[3 1 2]);

act=permute(max(XX.*WW,[],2),[1 3 2]);
%act=1./(1+exp(-(X*W+repmat(B,1,size(X,1))')));
%zeros(size(act))
%der=act.*(1-act);
der=ones(size(act));
end