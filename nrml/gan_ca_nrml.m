%% demo: NRML

clear all;
clc;
close all;

addpath('nrml');


kins = ['fd';'fs';'md';'ms'];
avg_value = zeros(4,1);
avg_AUC = zeros(4,1);
for k = 1:4

k_type = kins(k,:);
%% data & parametres
load(['data/gan_only/' k_type '_gfeature.mat']);


T = 1;        % Iterations
knn = 5;      % k-nearest neighbors
Wdims = 100;  % low dimension

un = unique(fold);
nfold = length(un);

%% NRML
t_sim = [];
t_ts_matches = [];
t_acc = zeros(nfold, 1);
for c = 1:nfold    
    trainMask = fold ~= c;
    testMask = fold == c;    
    tr_idxa = idxa(trainMask);
    tr_idxb = idxb(trainMask);
    tr_matches = matches(trainMask);    
    ts_idxa = idxa(testMask);
    ts_idxb = idxb(testMask);
    ts_matches = matches(testMask);
    
    %% do PCA  on training data
    X = ux;
    tr_Xa = X(tr_idxa, :);                  % training data
    tr_Xb = X(tr_idxb, :);                  % training data
    [eigvec, eigval, ~, sampleMean] = PCA([tr_Xa; tr_Xb]);
    Wdims = size(eigvec, 2);
    X = (bsxfun(@minus, X, sampleMean) * eigvec(:, 1:Wdims));

    tr_Xa_pos = X(tr_idxa(tr_matches), :);  % positive training data 
    tr_Xb_pos = X(tr_idxb(tr_matches), :);  % positive training data
    ts_Xa = X(ts_idxa, :);                  % testing data
    ts_Xb = X(ts_idxb, :);                  % testing data
    clear X;
    
    %% metric learning
    W = nrml_train(tr_Xa_pos, tr_Xb_pos, knn, Wdims, T); 
    
    ts_Xa = ts_Xa * W;
    ts_Xb = ts_Xb * W;
    
    %% cosine similarity
    sim = cos_sim(ts_Xa', ts_Xb');
    t_sim = [t_sim; sim(:)];
    t_ts_matches = [t_ts_matches; ts_matches];

    %% Accuracy
    [~, ~, ~, ~, acc] = ROCcurve(sim, ts_matches);
    t_acc(c) = acc;
    fprintf('Fold %d, Accuracy = %6.4f \n', c, acc);
end
fprintf('The mean accuracy = %6.4f\n', mean(t_acc));
avg_value(k) = mean(t_acc);
% %% plot ROC
% [fpr, tpr] = ROCcurve(t_sim, t_ts_matches);
% figure(1)
% plot(fpr, tpr);
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% grid on;
% 
% %%
%% plot ROC
[fpr, tpr,AUC] = ROCcurve(t_sim, t_ts_matches);
avg_AUC(k)=AUC;
% save(['data/AUC/' kins(k,:) '_wokmm.mat'],'fpr','tpr')
fprintf('%s AUC = %6.4f\n','fd',AUC)
figure
plot(fpr, tpr);
xlabel('False Positive Rate')
ylabel('True Positive Rate')
grid on;
end
fprintf('the AVG acc %6.4f\n',mean(avg_value))
fprintf('the AVG AUC %6.4f\n',mean(avg_AUC))
%%