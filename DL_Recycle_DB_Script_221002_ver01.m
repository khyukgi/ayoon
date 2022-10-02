clc; close all;
clear data R_DB_01 R_DB_02 R_DB_03;
%%
liste = dir('*m*.*');

for d=1:length(liste) % month
    folder = liste(d).name;
    pt = strcat('Loading... ','month: ',folder);
    disp(pt)
    %% analysis for each month
    DataPath = strcat('./',folder,'/');
    digitDatasetPath = fullfile(DataPath);
    imdsTest = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

    for i=1:50%length(imdsTest.Files)
        
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
%         h = figure; h.Position(3) = 3*h.Position(3);                
%         subplot(1,3,1);
%         imshow(rimg);
%         title(string(output) + ", " + num2str(100*max(score),3) + "%");
% 
%         subplot(1,3,2);
%         barh(scoreTop); xlim([0 1]);
%         title('Confidence Score');
%         xlabel('Probability');  yticklabels(classNamesTop); grid on;
% 
%         subplot(1,3,3); imshow(rimg);
%         hold on; imagesc(map,'AlphaData',0.5);
%         colormap jet; hold off;
%         title("Grad-CAM based Heat map");

%         h2 = figure; h2.Position(4) = h2.Position(4);
%         subplot(3,1,1); bar(pixel_level, red_data,'r');
%         subplot(3,1,2); bar(pixel_level, green_data,'g');
%         subplot(3,1,3); bar(pixel_level, blue_data,'b');
        
        %% construct Recycle-DB(databased)
        data(i,1) = classfn;
        if (data(i,1) == "pet_pet")
            data(i,2) = '1';
            data(i,3) = 'Recyclable';
            data(i,4) = '1';
        elseif (data(i,1) == "drink_can")
            data(i,2) = '2';
            data(i,3) = 'Recyclable';
            data(i,4) = '1';
        elseif (data(i,1) == "snack_vinyl")
            data(i,2) = '3';
            data(i,3) = 'NOT Recyclable';
            data(i,4) = '2';
        end
    end
    %% Set R-DB

    % count trash category
    R_DB_01(d,1) = sum(data(:,1) == "pet_pet");
    R_DB_01(d,2) = sum(data(:,1) == "drink_can");
    R_DB_01(d,3) = sum(data(:,1) == "snack_vinyl");

    % count recyclable (Y/N)
    R_DB_02(d,1) = sum(data(:,3) == "NOT Recyclable");
    R_DB_02(d,2) = sum(data(:,3) == "Recyclable"); 
    
    % color
    R_DB_03(d,1) = max(red_data(:));
    R_DB_03(d,2) = max(green_data(:));
    R_DB_03(d,3) = max(blue_data(:));     
end
