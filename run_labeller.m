clc;clear all;close all force;


% path = 'D:\nove_skluzavky_save\nuc_for_labeling\sample1';

% filenames_dapi = subdir([path '/*_DAPI.tiff']);
% rng(42)
% perm = randperm(length(filenames_dapi));
% filenames_dapi_perm = {filenames_dapi(perm).name};
% save('filenames_dapi_perm.mat','filenames_dapi_perm')

load('filenames_dapi_perm.mat','filenames_dapi_perm')



% path = 'D:\nove_skluzavky_save\nuc_for_labeling\sample2';
% 
% filenames_dapi = subdir([path '/*_DAPI.tiff']);
% rng(42)
% perm = randperm(length(filenames_dapi));
% filenames_dapi_perm = {filenames_dapi(perm).name};
% save('filenames_dapi_perm_sample2.mat','filenames_dapi_perm')

% load('filenames_dapi_perm_sample2.mat','filenames_dapi_perm')


for file_num = 1:432

    

    disp(['file_num ' num2str(file_num)])

    filename_dapi = filenames_dapi_perm{file_num};
    filename_qpi = replace(filename_dapi,'_DAPI.tiff','_QPI.tiff');
    filename_save = [replace(filename_dapi,'_DAPI.tiff','') '_detections.mat'];

%     if contains(filename_dapi,'Citronellol_2mM_PC3_DPC__p06_t261_DAPI.tiff')
% 
%     elseif contains(filename_dapi,'FaDu_Matrine_z_VAD_Fluo__p18_t226_DAPI.tiff')
% 
%     elseif contains(filename_dapi,'FaDu_RSL3_10uM_zVADFMK_FLUO__p14_t197_DAPI.tiff')
% 
%     elseif contains(filename_dapi,'FaDu_Matrine_z_VAD_Fluo__p17_t089_DAPI.tiff')
% 
%     elseif contains(filename_dapi,'FaDu_Matrine_z_VAD_Fluo__p21_t266_DAPI.tiff')
% 
%     elseif contains(filename_dapi,'FaDu_Matrine_z_VAD_Fluo__p15_t259_DAPI.tiff')
% 
%     elseif contains(filename_dapi,'FaDu_Matrine_z_VAD_Fluo__p19_t098_DAPI.tiff')
% 
%     else
%         continue
%     end

    

    dapi = imread(filename_dapi);
    qpi = imread(filename_qpi);
    
%     tmp_qpi = medfilt2(qpi,[7,7]);
%     tmp_qpi = imgaussfilt(tmp_qpi,7);
% 
%     tmp = medfilt2(dapi,[5,5]);
%     tmp = imgaussfilt(tmp,3);
% 
%     tmp = imhmax(tmp,3);
%     tmp = imdilate(tmp,strel('disk',3));
%     tmp2 = imregionalmax(tmp);
%     props = regionprops(tmp2,tmp,'Centroid','MaxIntensity');
%     centroid = cat(1,props.Centroid);
%     value = [props.MaxIntensity];
% 
%     props = regionprops(tmp2,tmp_qpi,'MaxIntensity');
%     value_qpi = [props.MaxIntensity];
% 
%     detections = centroid(value>105 & value_qpi>0.1 ,:);


    filename_prediction = replace(filename_dapi,'_DAPI.tiff','_prediction.tiff');
    filename_prediction = replace(filename_prediction,'nuc_for_labeling','nuc_prediction2');

    prediction = imread(filename_prediction);
    tmp = prediction;

    tmp = imhmax(tmp,0.03);
    tmp = imdilate(tmp,strel('disk',3));
    tmp2 = imregionalmax(tmp);
    props = regionprops(tmp2,tmp,'Centroid','MaxIntensity');
    centroid = cat(1,props.Centroid);
    value = [props.MaxIntensity];

    detections = centroid(value>0.1 ,:);


%     figure();
%     imshow(prediction,[])
%     hold on;
%     plot(detections(:,1),detections(:,2),'*')
% 
% 
%     figure();
%     imshow(dapi,[])
%     hold on;
%     plot(detections(:,1),detections(:,2),'*')
%     dsdfds




%     imshow(dapi,[])
%     hold on;
%     plot(detections(:,1),detections(:,2),'*')
%     dsdfds


    app = nuc_labeller(dapi, qpi, detections, file_num);
    while ~app.done
        pause(0.1); 
    end
    detections = app.detections;
    app.delete;
    

    save(filename_save,'detections')

end


