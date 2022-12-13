function [mask8] = conn4to8(mask4)

% mask4 = imread("C:\Users\vicar\Desktop\kody\nove_skluzavky_labeler\nove_skluzavky_save\predicted2\Citronellol_2mM_PC3_DPC__p00_t192_predicted_mask.tif");
L = bwlabel(mask4>0,4);

overlap = zeros(size(mask4));


tri = [0 1 0; 0 1 1; 0 0 0];
cross = [0 1 0; 1 1 1; 0 1 0];
square = [1 1 1; 1 1 1; 1 1 1];
xcross = [1 0 1; 0 1 0; 1 0 1];
xtri = [1 0 1; 0 1 0; 0 0 0];

N = max(L(:));
for lable_num = 1:N
    overlap = overlap + imdilate(L == lable_num, xtri);
end
mask8 = mask4;
mask8(overlap>1) = 0;

% imshow(double(mask4>0) + 5*double(overlap>1),[])
% end

% imshow(10*double(overlap>1),[])

% imshow(mask8,[])
