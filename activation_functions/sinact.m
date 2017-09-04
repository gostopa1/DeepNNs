function [act,der]=sinact(X,W,B)
%preact=(X*W+repmat(B,1,size(X,1))');
%act=sin(preact*2*pi*100);
%der=cos(preact*2*pi*100);


preact=(X*W);
act=sin(preact*2*pi*100.*repmat(B,1,size(X,1))');
der=cos(preact*2*pi*100.*repmat(B,1,size(X,1))');

end