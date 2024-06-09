
% clear; %clc;
addpath(genpath('./.'));
run('C:\matconvnet-1.0-beta24\matconvnet-1.0-beta24\matlab\vl_setupnn.m');

%% testing set
imageSets   = {'test','Set14','classic5'};
image_set   = imageSets{1};

folderTest = fullfile('Test_set',image_set);

showresult  = 1;
WF = 0;
gpu = 1;

if gpu 
    gpuDevice(gpu); 
end

modelName   = 'MWCNN_deblurring_DBT6-epoch-20';%;'MWCNN_Haart_GDSigma';

ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
end

fprintf('okundu. %d ',length(filePaths));
    

    %% load model
    load(fullfile('models', [modelName]));

    net = dagnn.DagNN.loadobj(net) ;   
    net.removeLayer('objective') ;
    out_idx = net.getVarIndex('prediction') ;
    net.vars(net.getVarIndex('prediction')).precious = 1 ;
    net.mode = 'test';  
    if gpu
        net.move('gpu');
    end
    for i = 1 : length(filePaths)
        %%% read images
        im = imread(fullfile(folderTest,filePaths(i).name));
        im  = modcrop(im, 8);
        if size(im,3)==3
            label_im = rgb2gray(im);
            label = label_im(:,:,1);
        else
            label = im;
        end
        sz = size(label);
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        label = im2single(label);

        input = label; % Blur the image.
        
        tic;
        output = Processing_Im(input, net, gpu, out_idx);
        times(i) = toc;
        
        %imshow(output);
         if showresult
            imshow(cat(2,im2uint8(input),im2uint8(output),im2uint8(label)));
            title([filePaths(i).name])
            drawnow;
        end
        if WF 
            path =  ['./results/' modelName '/' image_set '_Sigma' num2str(Sigma)];
            if ~exist(path, 'dir'), mkdir(path) ; end
            imwrite(output, fullfile(path, [modelName '-' num2str(epoch) '-' filePaths(i).name]));
        end
        
        image_name = filePaths(i).name;
        imwrite(output, filePaths(i).name);
    end
    if WF 
        save(fullfile(path, [modelName '_' image_set  '_Sigma' num2str(Sigma) 'PSNR']), 'PSNRs');
        save(fullfile(path, [modelName '_' image_set  '_Sigma' num2str(Sigma) 'SSIM']), 'SSIMs');
    end
    








