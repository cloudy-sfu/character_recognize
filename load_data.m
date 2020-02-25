clear; close all;
load emnist-balanced.mat

X_train = dataset.train.images;
Y_train = dataset.train.labels;
X_test = dataset.test.images;
Y_test = dataset.test.labels;
[num_train, ~] = size(X_train);
[num_test, ~] = size(X_test);

X_train = reshape(X_train, [num_train, 28, 28]);
X_test = reshape(X_test, [num_test, 28, 28]);
save('py_emnist_balanced.mat','X_train','X_test',...
    'Y_train', 'Y_test', '-v7');