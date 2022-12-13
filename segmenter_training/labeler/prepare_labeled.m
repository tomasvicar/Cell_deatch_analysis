clc;clear all;close all;
addpath('utils')


data_path='C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\nuc_for_labeling\sample3';
save_path='C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\labeled_cells1\';


names=subdir([data_path '/*.png']);

names={names(:).name};




for img_num = 1:length(names)
    name_mask=names{img_num};

    
    name_data=[name_mask(1:end-29) '.tiff'];

    [~,tmp,~] = fileparts(name_data);


    cell_line = 'cellDeath';

    name_data_save=[save_path '/' cell_line '/' tmp '_' cell_line '_img.tif'];
    name_mask_save=[save_path '/' cell_line '/' tmp '_' cell_line '_mask.png'];

    
    mkdir(fileparts(name_data_save))

    img = imread(name_data);

    lam = 0.65;
    alpha = 0.18;
    img = img * lam / (2 * pi * alpha) ;

    imwrite_single(img,name_data_save)


%     copyfile(name_data,name_data_save)
%     copyfile(name_mask,name_mask_save)
    mask=imread(name_mask);
    if length(size(mask)) > 2
        mask = mask(:,:,1);
    end

    mask = bwareaopen(mask>0,20,8);

    l=bwlabel(mask>0,8);
    
    imwrite(uint8(l),name_mask_save)
    

    
    
    
end



