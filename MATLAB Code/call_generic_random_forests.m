function call_generic_random_forests()
data = readtable('abalone.csv');
data.Properties.VariableNames = {'Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight','Rings'};
Y  = data.Sex;
X = removevars(data,{'Sex'});
BaggedEnsemble = generic_random_forests(X,Y,100,'classification');
tic
predict(BaggedEnsemble,[0.455 0.265 0.095 0.614 0.1245 0.161 0.125 12]);
toc
