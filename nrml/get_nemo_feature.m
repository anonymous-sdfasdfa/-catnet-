
clear;clear all;

ktype = ['father-dau';'father-son';'mother-dau';'mother-son'];

for k = 1:2
% % kintype = 'F-D';


ls_nm = ['/home/Documents/DATA/kinship/label/' kins(k,:) '.mat'];
img_pth = ['/home/Documents/DATA/kinship/' ktype(k,:)];
temp = load(ls_nm);
ls_f = temp.pairs;
N =  size(ls_f,1);
fold = zeros(N,1);
matches= zeros(N,1);
idxa = zeros(N,1);
idxb = zeros(N,1);

for i = 1: size(ls_f,1)
    if rem(i,500)==0
        fprintf('%d/%d%% \n',i,size(ls_f,1))
    end
    temfd = ls_f(i,1);
    fold(i) = temfd{1};
    temma = ls_f(i,2);
    matches(i) = logical(temma{1});
    temxa = ls_f(i,3);
    im1_num = cell2mat(ls_f(i,5));
    idxa(i)=im1_num;
    im1_pth = fullfile(img_pth,temxa{1});
    I = imread(im1_pth);
    I = imresize(I,[64,64]);
    I = rgb2gray(I);
    f_hog(im1_num,:) =double(extractHOGFeatures(I));
    f_lbp(im1_num,:) = double(get_batch_lbp(I));
    
    temxb = ls_f(i,4);
    im2_num = cell2mat(ls_f(i,6));
    idxb(i) = im2_num;
    im2_pth = fullfile(img_pth,temxb{1});
    I = imread(im2_pth);
    I = imresize(I,[64,64]);
    I = rgb2gray(I);
    f_hog(im2_num,:) = double(extractHOGFeatures(I));
    f_lbp(im2_num,:) =double(get_batch_lbp(I));
    
%     HOG_KII_FS.ux

    
end
matches = logical(matches);
ux = f_hog;
save(['data/basic_feature/HOG_' kins(k,:) '.mat'],'fold','idxa','idxb','matches','ux')
ux = f_lbp;
save(['data/basic_feature/LBP_' kins(k,:) '.mat'],'fold','idxa','idxb','matches','ux')
end



