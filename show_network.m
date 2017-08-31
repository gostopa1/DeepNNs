
figure(1)
clf
subplot(4,1,[1 2])
hold on
linefactor=2;
for layeri=1:(length(model.layersizes)-1)
    %for layeri=1:2
    
    ylocs1=(1:model.layersizes(layeri));
    ylocs1=ylocs1-mean(ylocs1);
    ylocs2=(1:model.layersizes(layeri+1));
    ylocs2=ylocs2-mean(ylocs2);
    
    
    for i = 1:model.layersizes(layeri)
        for j= 1:model.layersizes(layeri+1)
            if model.layers(layeri).W(i,j)~=0
                
                wid=abs(model.layers(layeri).W(i,j))*linefactor;
                if model.layers(layeri).W(i,j)>0
                    col=[0 0 1];
                else
                    col=[1 0 0];
                end
                line([layeri layeri+1],[ylocs1(i) ylocs2(j)],'LineWidth',wid,'Color',col);
            end
        end
    end
end

%%

for layeri=1:length(model.layersizes)
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
    ylocs=(1:model.layersizes(layeri));
    ylocs=ylocs-mean(ylocs);
    for neuroni=1:model.layersizes(layeri)
        h=plot(layeri,ylocs(neuroni),'.k');
        set(h,'MarkerSize',50,'Color',nodecol);
    end
end
axis off
box off

%%
