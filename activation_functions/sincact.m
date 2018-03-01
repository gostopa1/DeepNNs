function [act,der]=softsign(X,W,B)
preact=X*W+repmat(B,1,size(X,1))';
act=zeros(size(preact));
act(preact~=0)=sin(preact(preact~=0))./preact(preact~=0);

der=zeros(size(preact));
der(preact~=0)=(cos(preact(preact~=0))./preact(preact~=0))-(sin(preact(preact~=0))./(preact(preact~=0).^2));

end