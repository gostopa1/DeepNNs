function [act,der]=softmaxact(X,W,B,target)
preact=X*W+repmat(B,1,size(X,1))';
exppreact=exp(preact);

act=exppreact./repmat(sum(exppreact,2),1,size(exppreact,2));

if ~sum(isnan(act))==0
    [~,inds]=max(preact');
    exppreact=ones(size(preact))*0.01;
    exppreact(:,inds)=0.99;
    act=exppreact./repmat(sum(exppreact,2),1,size(exppreact,2));
end

try
    der=(-act.*(target-act));
catch
    display('Number of samples do not match between inputs and targets! Cannot estimate derivative. Setting them to zero instead!')
    der=zeros(size(act));
end

uplim=0.99;
lowlim=0.01;
der(der>uplim)=uplim;
der(der<lowlim)=lowlim;
end