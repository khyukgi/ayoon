clc; close all; 

liste = dir('*m*.*');

d=1; % month
folder = liste(d).name;
pt = strcat('Loading... ','month: ',folder);
disp(pt)
% analysis for each month
DataPath = strcat('./',folder,'/');
digitDatasetPath = fullfile(DataPath);
imdsTest = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%i = 300;
%i = 1000;
%i = 1200;
%i = 1500;

i = 1513;

txt = strcat('Analyzing... No.', num2str(i));
disp(txt)

% load raw image
raw = imdsTest.Files(i);
label = imdsTest.Labels(i);

[img, cmap] = imread(char(raw));
%     imshow(img); title(string(label));

% 1) classify category using trained model
sz_img = size(img);% original size

rimg = imresize(img,inputSize(1:2));
[classfn,score] = classify(netTransfer,rimg);

% convert text-label
if (classfn == "pet_pet")
    output = 'PET- Recyclable';
elseif (classfn == "drink_can")
    output = 'CAN- Recyclable';
elseif (classfn == "snack_vinyl")
    output = 'Vinyl- Recyclable';
end

% ====================================================================================
% 2) plot confidence score for each catergory
% ====================================================================================
[~,idx] = sort(score,'descend');
idx = idx(3:-1:1);
classes = netTransfer.Layers(end).Classes;
classNamesTop = string(classes(idx));
scoreTop = score(idx);

% ====================================================================================
% 3) plot heat map using Grad-CAM
% ====================================================================================
map = gradCAM(netTransfer,rimg,classfn);

% ====================================================================================
% 4) feature extracted mask
% ====================================================================================
b_map = imbinarize(map);
% figure; imshow(b_map);

sz = size(rimg);
mimg = rimg .* uint8(b_map);
% figure; imshow(mimg, []);

% ====================================================================================
% 5) RGB-Histogram
% ====================================================================================
rmap = imresize(map,sz_img(1:2));
b_rmap = imbinarize(rmap);
%figure; imshow(b_rmap, []);

m_bimg = img .* uint8(b_rmap);

%he = m_bimg;
he = img;
%he = mimg;

[red_data pixel_level]=imhist(he(:,:,1));
[green_data pixel_level]=imhist(he(:,:,2));
[blue_data pixel_level]=imhist(he(:,:,3));

%% Plot results
% plot results- predicted image
h = figure; h.Position(3) = 3*h.Position(3);
subplot(1,3,1);
imshow(rimg);
title(string(output) + ", " + num2str(100*max(score),3) + "%");

subplot(1,3,2);
barh(scoreTop); xlim([0 1]);
title('Confidence Score');
xlabel('Probability');  yticklabels(classNamesTop); grid on;

subplot(1,3,3); imshow(rimg);
hold on; imagesc(map,'AlphaData',0.5);
colormap jet; hold off;
title("Grad-CAM based Heat map");

%         h2 = figure; h2.Position(4) = h2.Position(4);
%         subplot(3,1,1); bar(pixel_level, red_data,'r');
%         subplot(3,1,2); bar(pixel_level, green_data,'g');
%         subplot(3,1,3); bar(pixel_level, blue_data,'b');
