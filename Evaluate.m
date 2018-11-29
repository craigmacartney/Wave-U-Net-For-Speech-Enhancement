%% Matlab script for calculating speech enhancement metrics for multiple files
%composite.m for individual comparisons required. Available here: https://ecs.utdallas.edu/loizou/speech/composite.zip
%Metrics used: PESQ, CSIG, CBAK, COVL and SSNR

clc; clear; % clear command window and workspace

files = dir('./p*'); % directory to reference clean and enhanced speech files

f=zeros(824,3);
for k = 1:length(files)
   file = files(k).name;
   path=['./' file];
   [sig(k),bak(k),ovl(k)] = composite([path '/' file '_clean.wav'], [path '/' file '_speech_estimate.wav']);
   f(k,1) = sig(k);
   f(k,2) = bak(k);
   f(k,3) = ovl(k);
end
