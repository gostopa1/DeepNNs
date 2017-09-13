function impos=extract_LRP(model,x)
%% Output is samples x inputs x outputs

out=model.layers(end).out;
nosamples=size(out,1);

%W=model.layers(layeri).W;
%X=model.layers(layeri).X;
%B=model.layers(layeri).B;
noouts=model.layersizes(end);
for outi=1:noouts
    %layeri=length(model.layers);
    R=model.layers(length(model.layers)).out;
    
    R(:,setxor(1:noouts,outi))=0;
    
    %preacts=X*W+repmat(B,1,size(X,1))'
    for layeri=length(model.layers):-1:1
        
        %Now it implements simple W.^2 importance extraction
        
        W=model.layers(layeri).W;
        WWsR=permute(repmat((W.^2)./(repmat(sum(W.^2,1),model.layersizes(layeri),1)),1,1,nosamples),[3 1 2]);

        

        outsR=permute(repmat(R,1,1,model.layersizes(layeri)),[1 3 2]);
        
        
        R=sum(WWsR.*outsR,3);
        
    end
    
    impos(:,:,outi)=R;
end

end