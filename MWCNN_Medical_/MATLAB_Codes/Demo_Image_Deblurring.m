
% clear; %clc;
addpath(genpath('./.'));
run('C:\matconvnet-1.0-beta24\matconvnet-1.0-beta24\matlab\vl_setupnn.m');
%% testing set
imageSets   = {'solo','Set14','classic5'};
image_set   = imageSets{1};

folderTest = fullfile('testsets',image_set);
folderTest2 = fullfile('testsets2',image_set);

save_folder = 'C:\Users\Burak\Desktop\deblurring_MWCNN\MWCNN-master\examples';
showresult  = 1;
WF = 0;
gpu = 1;

if gpu 
    gpuDevice(gpu); 
end

list_sig = 20;%[1 10 20 30 40 45];
modelName   = 'MWCNN_deblurring6-epoch-';%;'MWCNN_Haart_GDSigma';

ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
    filePaths2 = cat(1,filePaths, dir(fullfile(folderTest2,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));
 

    
for Sigma = list_sig % [ 15 25 50
    %% load model
    load(fullfile('models', [modelName num2str(Sigma)]));

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

        %%
        randn('seed',0);
        
%         input = label + Sigma/255*randn(sz,'single');
%         windowWidth = 50; % Whatever you want.  More blur for larger numbers.
%         kernel = ones(windowWidth) / windowWidth ^ 2;
%         input = imfilter(label, kernel); % Blur the image.
%           input = imread(fullfile(folderTest2,filePaths2(i).name)); 

        %%% read image for input
        im2 = imread(fullfile(folderTest2,filePaths2(i).name));
        im2  = modcrop(im2, 8);
        if size(im2,3)==3
            label_im2 = rgb2gray(im2);
            label2 = label_im2(:,:,1);
        else
            label2 = im2;
        end
        sz = size(label2);
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        label2 = im2single(label2);
        input = label2;


        tic;
        output = Processing_Im(input, net, gpu, out_idx);
        times(i) = toc;
        

        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output), 0, 0);
        
%         figure;     
%         subplot(1,3,1), imshow(input), title('BLUR Image'); hold on
%         subplot(1,3,2), imshow(output), title('DEBLURED Image'); hold on
%         subplot(1,3,3), imshow(label), title('Ground Truth Image'); hold off
      
        if showresult
            imshow(cat(2,im2uint8(input),im2uint8(output),im2uint8(label)));
            title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f'), '   ' , 'Epoch Number = ', num2str(list_sig,'%d')])
            drawnow;
            imwrite(cat(2,im2uint8(input),im2uint8(output),im2uint8(label)), [num2str((i-1), '%02d') '.png']);
        end
        if WF 
            path =  ['./results/' modelName '/' image_set '_Sigma' num2str(Sigma)];
            if ~exist(path, 'dir'), mkdir(path) ; end
            imwrite(output, fullfile(path, [modelName '-' num2str(epoch) '-' filePaths(i).name]));
        end
        
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;
        fprintf('PSNR = %f SSIM = %f\n',PSNRCur,SSIMCur);
    end
    if WF 
        save(fullfile(path, [modelName '_' image_set  '_Sigma' num2str(Sigma) 'PSNR']), 'PSNRs');
        save(fullfile(path, [modelName '_' image_set  '_Sigma' num2str(Sigma) 'SSIM']), 'SSIMs');
    end
    fprintf('burdayiz...');
    fprintf('PSNR / SSIM : %.02f / %0.4f, %0.4f.\n', mean(PSNRs),mean(SSIMs), mean(times));
end   
    








