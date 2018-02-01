function [impos,imps_per_layer]=extract_LRPWX(model,x)
%% Output is samples x inputs x outputs
% The importances for each layer are also estimated if a second ouput
% argument is requested. The output is the model and the importance per
% layer are stored in model.layers(layer_number).R
[model,out]=forwardpassing(model,x);
nosamples=size(out,1);

%W=model.layers(layeri).W;
%X=model.layers(layeri).X;
%B=model.layers(layeri).B;
nosamples=size(x,1);
noouts=model.layersizes(end);
for outi=1:noouts
    %layeri=length(model.layers);
    R=model.layers(length(model.layers)).out;
    %R=model.layers(length(model.layers)).out.*(model.layers(end).X*model.layers(end).W+repmat(model.layers(end).B,1,size(x,1))');;
    
    R(:,setxor(1:noouts,outi))=0;
    
    %preacts=X*W+repmat(B,1,size(X,1))'
    layeri=1;
    for layeri=length(model.layers):-1:1
        n=model.layersizes(layeri+1);
        m=model.layersizes(layeri);
        
        %Now it implements simple W.^2 importance extraction
        Wr=permute(repmat(model.layers(layeri).W,1,1,nosamples),[3 1 2]);
        Xr=repmat(model.layers(layeri).X,1,1,n);
        Br=permute(repmat(model.layers(layeri).B,1,model.layersizes(layeri),nosamples),[3 2 1]);
        preact=(Wr.*Xr+Br);
        Z=Wr.*Xr;
        Zs=sum(Z,2);
        Rr=repmat(permute(R,[1 3 2]),1,m,1);
        R=sum((Z./repmat(Zs,1,m,1)).*Rr,3);
        
        %model.layers(layeri).R(:,:,outi)=R;
        imps_per_layer(layeri).R=R;
    end
    
    
    impos(:,:,outi)=R;
    %impos=model.layers(layeri).R(:,:,outi);
end

end