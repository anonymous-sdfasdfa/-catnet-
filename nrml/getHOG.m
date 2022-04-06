kins = ['fd';'fs';'md';'ms'];
ktype = ['father-dau','father-son','mother-dau','mother-son'];
for k = 1:4
    
ls_nm = ['D:\NRML\KinFaceW-I\meta_data\' kins(k,:) '_pairs.mat'];
img_pth = ['D:\NRML\KinFaceW-I\images\' ktype(k,:)];
temp = load(ls_nm);
ls_f = temp.pairs;
N =  size(ls_f,1);
fold = zeros(N,1);
matches= zeros(N,1);
idxa = zeros(N,1);
idxb = zeros(N,1);

for i = 1: size(ls_f,1)
    fprintf('%d/%d%% \n',i,size(ls_f,1))
    temfd = ls_f(i,1);
    fold(i) = temfd{1};
    temma = ls_f(i,2);
    matches(i) = logical(temma{1});
    temxa = ls_f(i,3);
    im1_num = (str2double(temxa{1}(4:6))-1)*2+str2double(temxa{1}(8));
    idxa(i)=im1_num;
    im1_pth = fullfile(img_pth,temxa{1});
    I = imread(im1_pth);
    I = rgb2gray(I);
    f_hog(im1_num,:) =double(extractHOGFeatures(I));
    f_lbp(im1_num,:) = double(get_batch_lbp(I));
    
    temxb = ls_f(i,4);
    im2_num = (str2double(temxb{1}(4:6))-1)*2+str2double(temxb{1}(8));
    idxb(i) = im2_num;
    im2_pth = fullfile(img_pth,temxb{1});
    I = imread(im2_pth);
    I = rgb2gray(I);
    f_hog(im2_num,:) = double(extractHOGFeatures(I));
    f_lbp(im2_num,:) =double(get_batch_lbp(I));
    
%     HOG_KII_FS.ux

    
end
matches = logical(matches);
ux = f_hog;
save(['HOG_KII_' kin(k,:) '.mat'],'fold','idxa','idxb','matches','ux')
ux = f_lbp;
save('LBP_KII_' kin(k,)'.mat','fold','idxa','idxb','matches','ux')

end


