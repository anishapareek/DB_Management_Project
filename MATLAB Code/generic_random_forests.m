function BaggedEnsemble = generic_random_forests(X,Y,iNumBags,str_method)


BaggedEnsemble = TreeBagger(iNumBags,X,Y,'OOBPred','On','Method',str_method)

% plot out of bag prediction error
oobErrorBaggedEnsemble = oobError(BaggedEnsemble);
figID = figure;
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
print(figID, '-dpdf', sprintf('randomforest_errorplot_%s.pdf', date));

oobPredict(BaggedEnsemble)

% % view trees
view(BaggedEnsemble.Trees{1}) % text description
view(BaggedEnsemble.Trees{1},'mode','graph') % graphic description
