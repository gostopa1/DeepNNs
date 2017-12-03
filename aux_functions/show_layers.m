function show_layers(model, layer)
% Function that shows specific layers of the model (can show all if all layers are selected)
% The NN model is the first argument, the layer argument is an int or a
% vector of the layers to be visualized

hold on
linefactor=2;
for layeri=layer
     ylocs1=(1:model.layersizesinitial(layeri));
    
    ylocs1=ylocs1-mean(ylocs1);
    
    ylocs2=(1:model.layersizesinitial(layeri+1));
    
    ylocs2=ylocs2-mean(ylocs2);
    
    ylocs1=ylocs1(model.layers(layeri).inds);
    if layeri<length(model.layers)
    ylocs2=ylocs2(model.layers(layeri+1).inds);
    end
    
    
    for i = 1:model.layersizes(layeri)
        for j= 1:model.layersizes(layeri+1)
            if model.layers(layeri).W(i,j)~=0
                
                wid=abs(model.layers(layeri).W(i,j))*linefactor;
                if model.layers(layeri).W(i,j)>0
                    col=[0 0 1];
                else
                    col=[1 0 0];
                end
                try 
                    line([layeri layeri+1],[ylocs1(i) ylocs2(j)],'LineWidth',wid,'Color',col);
                catch
                    line([layeri layeri+1],[ylocs1(i) ylocs2(j)],'LineWidth',2,'Color','Black','LineStyle','--');
                end
            end
        end
    end
end

%%

for layeri=layer:(layeri+1)
    if layeri>1
        switch model.layers(layeri-1).activation
            case 'logsi'
                nodecol=[0.5 0.5 1];
            case 'tanhact'
                nodecol=[1 0.5 0.5];
            case 'relu'
                nodecol=[1 0.5 0.5];
            case 'linact'
                nodecol=[0.8 0.8 0.8];
            case 'softmaxact'
                nodecol=[0.5 1 0.5];
            otherwise,
                nodecol=[0 0 0];
        end
    else
        nodecol=[0 0 0.5];
        
    end
    ylocs=(1:model.layersizesinitial(layeri));
    ylocs=ylocs-mean(ylocs);
    
    if layeri<=length(model.layers)
    ylocs=ylocs(model.layers(layeri).inds);
    end
    for neuroni=1:model.layersizes(layeri)
        h=plot(layeri,ylocs(neuroni),'.k');
        set(h,'MarkerSize',50,'Color',nodecol);
    end
end
ylims=minmax(1:max(model.layersizesinitial))-mean(1:max(model.layersizesinitial));
axis([1 length(model.layers)+1 ylims(1)-0.5 ylims(2)+0.5])
axis off
box off
end