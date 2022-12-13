clc;clear all;close all;


name_det = "D:\nove_skluzavky_save\nuc_for_labeling\sample1\FaDu_RSL3_FLUO__p19_t219_detections.mat";
name_dapi = replace(name_det,'_detections.mat','_DAPI.tiff');


load(name_det,'detections')
dapi = imread(name_dapi);
imshow(dapi,[])
hold on;
plot(detections(:,1),detections(:,2),'*')