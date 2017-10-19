function perf=get_perf(out_test,y_test)
%% This function measures the classification accuracy in terms of correct guesses. 
%% The inputs are the output of the classifier and the correct labels, both in matrix format (i.e. number of outputs x samples)
[~,indpre]=max(out_test(:,:,end)');
[~,indtarget]=max(y_test');
perf=(sum(indpre==indtarget)/length(indpre))*100;
end