
name_mask = 'C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\nuc_for_labeling\sample3\FaDu_Doxo_fluo_pokracovani__p17_t134_QPI.tiff';
name_mask = replace(name_mask,'_QPI.tiff','_mask_noremove.tiff');
name_mask = replace(name_mask,'nuc_for_labeling','cell_prediction_1');
mask = imread(name_mask);
imshow(mask,[])