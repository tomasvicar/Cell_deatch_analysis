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


for img_num = 1:20

    save(num2str(img_num),'img_num')
    
    img_name=files{img_num};
    img = imread(img_name);


%     name_mask = replace(img_name,'_QPI.tiff','_mask_noremove.tiff');
%     name_mask = replace(name_mask,'nuc_for_labeling','cell_prediction_1');

    labeled_name = subdir(replace(img_name,'.tiff','*.png'));
    labeled_name = labeled_name(1).name;
    labeled = imread(labeled_name)/25;

    figure()
    imshow(img,[])
    hold on
    colormap_cells=[1 0 0;0 1 0;0 0 1;0.8314 0.8314 0.0588;1 0 1;1,0.5,0;0.00,1.00,1.00;0.45,0.00,0.08];
    for k=1:8
        visboundaries(labeled==k,'Color',colormap_cells(k,:),'EnhanceVisibility',0,'LineWidth',0.1);
    end
    [~,tmp] = fileparts(labeled_name);
    title(tmp)

%     app=segmentation_tool(img_name, kdo_to_klika);



end











