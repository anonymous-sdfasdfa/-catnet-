clear; close all;
kins = ['fd';'fs';'md';'ms'];

k = 4
    
catnorm = load(['data/AUC-kfw1/' kins(k,:) '_IIATNet_kmm.mat']);



age0 = load(['data/AUC-kfw1/' kins(k,:) '_IIATNet_kmm_age.mat']);



hog = load(['data/AUC-kfw1/' kins(k,:) '_auc_hog.mat']);



lbp = load(['data/AUC-kfw1/' kins(k,:) '_auc_lbp.mat']);



lbphog = load(['data/AUC-kfw1/' kins(k,:) '_auc_lbphog.mat']);

wokmm = load(['data/AUC-kfw1/' kins(k,:) '_wokmm.mat']);

figure
plot(lbp.fpr,lbp.tpr,'b', hog.fpr,hog.tpr,'c',lbphog.fpr,lbphog.tpr,'k:', wokmm.fpr,wokmm.tpr,'m-.', age0.fpr, age0.tpr,'g--',catnorm.fpr, catnorm.tpr,'r','LineWidth',2);
legend({'LBP','HOG','LBP+HOG','w/o KMM','w/o aging','MIIA'},'Location','northwest')
% plot(lbp.fpr,lbp.tpr,'b', hog.fpr,hog.tpr,'c',lbphog.fpr,lbphog.tpr,'k:', wokmm.fpr,wokmm.tpr,'m-.',catnorm.fpr, catnorm.tpr,'r','LineWidth',2);
% legend({'LBP','HOG','LBP+HOG','w/o KMM','mCAT'},'Location','northwest')

xlabel('False Positive Rate')
ylabel('True Positive Rate')
grid on;

