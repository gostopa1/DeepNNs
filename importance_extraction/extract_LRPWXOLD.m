function [impos,model,Rc]=extract_LRPWXOLD2(model,x)
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
N=nosamples;
ypred=model.layers(length(model.layers)).out;
for outi=1:noouts
    
    ypred=model.layers(length(model.layers)).out;
    ypred(:,setxor(1:noouts,outi))=0;
    R=ypred;

    
    
    for layeri=length(model.layers):-1:1
        R=R.*(model.layers(layeri).X*model.layers(layeri).W+repmat(model.layers(layeri).B,1,size(x,1))');    
        n=model.layersizes(layeri+1);
        m=model.layersizes(layeri);
        
        %Now it implements simple W.^2 importance extraction
        Wr=permute(repmat(model.layers(layeri).W,1,1,nosamples),[3 1 2]);
        Xr=repmat(model.layers(layeri).X,1,1,model.layersizes(layeri+1));
        
        Z=Wr.*Xr;
        Zs=sum(Z,2) + repmat(permute(model.layers(layeri).B,[3 2 1]),[nosamples,1,1]);
        Rr=repmat(permute(R,[1 3 2]),1,model.layersizes(layeri),1);
        R=sum((Z./repmat(Zs,1,model.layersizes(layeri),1)).*Rr,3);
        
        model.layers(layeri).R(:,:,outi)=R;
    end
    
    
    impos(:,:,outi)=R;
    
end

end