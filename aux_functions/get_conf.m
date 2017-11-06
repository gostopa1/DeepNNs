function confmat=get_conf(out_test,y_test)
%% This function measures the classification accuracy in terms of correct guesses. 
%% The inputs are the output of the classifier and the correct labels, both in matrix format (i.e. number of outputs x samples)

[~,indpre]=max(out_test(:,:,end)');
[~,indtarget]=max(y_test');

nocats=max(indtarget);
confmat=zeros(nocats);

for i=1:length(indpre)
    confmat(indpre(i),indtarget(i))=confmat(indpre(i),indtarget(i))+1;
end

end