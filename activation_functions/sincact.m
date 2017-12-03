function [act,der]=softsign(X,W,B)
preact=X*W+repmat(B,1,size(X,1))';
act=zeros(size(preact));
act(preact~=0)=sin(preact)./preact;

der=zeros(size(preact));
der(preact~=0)=(cos(preact)./preact)-(sin(preact)./(preact.^2));

end