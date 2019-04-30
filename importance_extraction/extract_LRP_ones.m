function [impos,imps_per_layer]=extract_LRP_ones(model,x)
% This estimates the importances (or relevance) of the inputs, as well as
% the importances for each node (second arguments) based on the article by
% Montavon et al., 2017
% This is based on back-propagating the output to the input nodes.
% The current implementation is based on the squared weights
% % The importances are calculated for each sample in x
% Output is samples x inputs x outputs
% The importances for each layer are also estimated if a second ouput
% For the second output argument, importance per
% layer are stored in model.layers(layer_number).R
%
% The amount that will be backpropagated equals to the output of the
% classifier for each input and each class. Alternatively it can be set to
% one so that the magnitude of the importances is independent of the
% classifier's certainty (done in line 22).
[model,out]=forwardpassing(model,x);
nosamples=size(x,1);
noouts=model.layersizes(end);
for outi=1:noouts
    %R=model.layers(length(model.layers)).out;
    R=model.layers(length(model.layers)).out*0+1;    
    R(:,setxor(1:noouts,outi))=0;
    %R=mean(R,1); nosamples=1;
    for layeri=length(model.layers):-1:1
        n=model.layersizes(layeri+1);
        m=model.layersizes(layeri);
        
        %Now it implements simple W.^2 importance extraction
        
        %Wr=permute(repmat(model.layers(layeri).W,1,1,nosamples),[3 1 2]);
        %Xr=repmat(model.layers(layeri).X,1,1,n);
        %Br=permute(repmat(model.layers(layeri).B,1,model.layersizes(layeri),nosamples),[3 2 1]);
        %preact=(Wr.*Xr+Br);
        %Z=Wr.*Xr;
        Z=permute(repmat(full(model.layers(layeri).W.^2),1,1,nosamples),[3 1 2]);
        Zs=sum(Z,2);
        Rr=repmat(permute(R,[1 3 2]),1,m,1);
        R=sum((Z./repmat(Zs,1,m,1)).*Rr,3);

        imps_per_layer(layeri).R(:,outi,:)=R';
    end
    impos(:,outi,:)=R';
    
end

end
