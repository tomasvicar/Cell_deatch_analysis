clc;clear all;close all force;
addpath('utils')

kdo_to_klika = 'vicar';


%%%nastavit
% folder='C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\nuc_for_labeling\sample3';
% 
% 
% 
% files=subdir([folder filesep '*_QPI.tiff']);
% files={files(:).name};
% files = files(66:end);%% prvnich 65 sem udelal nenahodne
% 
% rng(42)
% perm = randperm(length(files));
% files = files(perm);
% save('random_filenames_65end.mat','files')


load('random_filenames_65end.mat','files')


for img_num = 139:140

    
    
    img_name=files{img_num};



    app=segmentation_tool(img_name, kdo_to_klika);

    save(num2str(img_num),'img_num')
    
end











