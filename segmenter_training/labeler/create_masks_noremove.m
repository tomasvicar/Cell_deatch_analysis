clc;clear all;close all force;
addpath('utils')


files=subdir('C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\nuc_for_labeling\sample3\*QPI.tiff');
files={files(:).name};

for file_num = 1:length(files)
    file_name = files{file_num};

    file_name_fg = replace(replace(file_name, '_QPI.tiff','_prediction_fg.tiff'), 'nuc_for_labeling', 'cell_prediction_1');

    file_name_dt = replace(replace(file_name, '_QPI.tiff','_prediction_dt.tiff'), 'nuc_for_labeling', 'cell_prediction_1');

    file_name_detection = replace(replace(file_name, '_QPI.tiff','_detections.json'), 'nuc_for_labeling', 'nuc_detection');

    file_name_nuc_pred = replace(replace(file_name, '_QPI.tiff','_prediction.tiff'), 'nuc_for_labeling', 'nuc_prediction3');

% 
%     file_name_save = [replace(replace(file_name, '_QPI.tiff',''), 'nuc_for_labeling', 'cell_prediction_1') '_mask.tiff'];
    file_name_save = [replace(replace(file_name, '_QPI.tiff',''), 'nuc_for_labeling', 'cell_prediction_1') '_mask_noremove.tiff'];

%     if ~strcmp(file_name_save,'C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\cell_prediction_1\sample3\FaDu_Doxo_fluo_pokracovani__p17_t134_mask_noremove.tiff')
%         continue;
%     end

    img = imread(file_name);
    fg = imread(file_name_fg);
    dt = imread(file_name_dt);
    nuc_pred = imread(file_name_nuc_pred);

    detection = jsondecode(fileread(file_name_detection));
    detection = detection.("detection");

    detections_bin = zeros(size(img));
    detections_bin(sub2ind(size(img),detection(:,2),detection(:,1))) = 1;


    tmp2 = fg;
    tmp3 = bwlabel(fg);
    s = regionprops(tmp3, detections_bin,"MaxIntensity");
    max_int = s.MaxIntensity;
    for k = 1:length(max_int)
        if max_int == 0
            tmp2(tmp3 ==k) = 0;
        end
    end
    filtered = fg - tmp2;
    fg = tmp2;



    tmp = -dt;
    tmp = imimposemin(tmp, detections_bin);
    tmp(~fg) = 999;
    tmp = watershed(tmp) > 0;
    tmp(~fg) = 0;

    
    tmp(filtered>0) = 1;


    tmp2 = bwareafilt(tmp, [80,Inf], 8);

    filtered2 = tmp - tmp2;
    tmp = tmp2;


    cross = [0 1 0; 1 1 1; 0 1 0];

%     filtered_L = bwlabel(filtered | filtered2);
    filtered_L = bwlabel(filtered2);
    N = max(filtered_L(:));
    for k = 1:N
        bin = filtered_L == k;
        bin_dil = imdilate(bin,cross);
        L = bwlabel(tmp,8);
        L_dil = imdilate(L,cross);

        if sum(bin_dil .* L_dil, 'all') > 0

            L_num = bin_dil .* L_dil;
            L_num = L_num(L_num>0);
            L_num = mode(L_num, 'all');

            tmp(bin) = 1;
            tmp(((L_dil ==L_num) .* bin_dil) > 0) = 1;
        end
    end

    tmp = ~bwareafilt(~tmp,[100,Inf], 8);

%     imshow(tmp,[])
% 
%     dfdsfdsdf


    imwrite(tmp, file_name_save)


end