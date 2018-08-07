function [act,der]=softmaxact(X,W,B)

preact=X*W+repmat(B,1,size(X,1))';

exppreact=exp(preact);

act=exppreact./repmat(sum(exppreact,2),1, size(exppreact,2));

upthres=0.99; downthres=0.01;

act(act>upthres)=upthres;
act(act<downthres)=downthres;

try
    
    der=act.*(1-act); % This corresponds to the diagonal of the Jacobian
    %der=ones(size(act)); % Assigning the derivatives to ones, seems to work better without a sound explanation. Otherwise, when using softmax and cross-entropy, the product of their derivatives (i.e. dE/net=dE/dout * dout/dnet) is always positive, that makes the weights in the output layer to grow infinitely and explode.
catch
    display('Number of samples do not match between inputs and targets! Cannot estimate derivative. Setting them to zero instead!')
    der=zeros(size(act));
end



end