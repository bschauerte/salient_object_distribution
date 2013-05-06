% ANALYZE_CENTER_BIAS_ACHANTA statistically analyzes the centroid
%   distribution of salient objects in Achanta's salient object detection
%   data set.
%
% Please see [1] for details and an application of the findings. Be so kind
% to cite [1], if you use the provided code.
%
%   [1] B. Schauerte, R. Stiefelhagen, "How the Distribution of Salient
%       Objects in Images Influences Salient Object Detection". In Proceedings
%       of the 20th International Conference on Image Processing (ICIP), 2013.
%
% @author: B. Schauerte
% @date:   2012-2013
% @url:    http://cvhci.anthropomatik.kit.edu/~bschauer/

% Copyright 2012-2013 B. Schauerte. All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%    1. Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
% 
%    2. Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the 
%       distribution.
% 
% THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
% IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
% DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
% BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
% WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
% OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
% ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% The views and conclusions contained in the software and documentation
% are those of the authors and should not be interpreted as representing 
% official policies, either expressed or implied, of B. Schauerte.

%% (optional) libraries
if isempty(which('randraw')), addpath(genpath('../libs/randraw/')); end
if isempty(which('circ_rtest')), addpath(genpath('../libs/circstat/')); end

%% Set path to Achanta's dataset (we only need the binary masks)
if ~exist('maskpath','var')
  if ispc
    maskpath='C:\Users\Boris\Desktop\data\achanta\binarymasks\';
  elseif ismac
    maskpath='/Users/bschauer/data/salient-objects/achanta/binarymasks/';
  else
    maskpath='/home/bschauer/data/salient-objects/achanta/binarymasks/'; % from Achanta
  end
end

%% Download the data, if necessary
if ~exist(maskpath)
  maskpath = 'binarymasks'; % did we, maybe, already download and locally setup a copy of the dataset?
end
if ~exist(maskpath)
  urlwrite('http://cvhci.anthropomatik.kit.edu/~bschauer/datasets/achanta_binary_masks.zip','binarymasks.zip');
  try
    unzip('binarymasks.zip');
  catch err
    % @NOTE: If you have trouble unzipping, then you might want to upgrade
    %        the (un-)zip command. The package is fine, but some versions
    %        of unzip seem to have trouble (Matlab 2012b on Mac OS X for
    %        example)
    error('Could not find or downloads Achanta''s dataset.')
  end
end

%% initialize data structures, counters & co.
num_images          = 0;
total_num_images    = 1000;
count_num_centroids = 0;
total_num_centroids = 1265;

mean_image=[];
image_sizes=zeros(total_num_images,2);

centroids=zeros(1265,2);
radius_centroids=zeros(1265,1);
radius_centroids_normalized=zeros(1265,1);
angles_centroids=zeros(1265,1);

%% calculate all centroids
maskfiles=dir(maskpath);
time_total_start=tic;
for i=1:numel(maskfiles)
  maskpathfull=[maskpath maskfiles(i).name];
  
  if length(maskfiles(i).name) < 5
    continue; % simple way to skip . and ..
  end
  
  num_images = num_images + 1;
  
  filename=maskfiles(i).name(1:end-4); % maskname w/o file ending
  tmp=sscanf(filename,'%d_%d_%d');
  
  mask_img=imread(maskpathfull);
  if isempty(mean_image)
    mean_image = double(imresize(mask_img,[325 372],'nearest'));
  else
    mean_image = mean_image + double(imresize(mask_img,[325 372],'nearest'));
  end
  
  % ... estimate the center of mass ...
  s = regionprops(bwlabel(sum(mask_img,3),8)); % determine and mark the connected components,'Orientation','Centroid');
  num_centroids = length(s);
  for k = 1:num_centroids
    count_num_centroids = count_num_centroids + 1;
    x0 = s(k).Centroid(1);
    y0 = s(k).Centroid(2);
    radius_centroids(count_num_centroids)=sqrt((x0 - size(mask_img,2)/2).^2 + (y0 - size(mask_img,1)/2).^2);
    angles_centroids(count_num_centroids)=atan2(y0 - size(mask_img,1)/2,x0 - size(mask_img,2)/2);
    radius_centroids_normalized(count_num_centroids)=sqrt((x0/size(mask_img,2) - 1/2).^2 + (y0/size(mask_img,1) - 1/2).^2);
    centroids(count_num_centroids,:) = [(x0 / size(mask_img,2)) (y0 / size(mask_img,1))];
  end
      
  image_sizes(num_images,:) = [size(mask_img,1) size(mask_img,2)];
end

assert(num_images == total_num_images); % if this fails, then bwlabel's implementation changed/differs from mine (i.e., the reference)

%% salient object mask and centroid visualizations

% mean image
mean_image = mean_image / num_images;
figure('name','mean segmentation mask'), 
imshow(mat2gray(mean_image))

% centroids
mean_centroids = mean(centroids);
stddev_centroids = std(centroids);
fprintf('centroids: mean=[%f %f] stddev=[%f %f]\n',mean_centroids(1),mean_centroids(2),stddev_centroids(1),stddev_centroids(2));
figure('name','centroid locations')
scatter(centroids(:,1),centroids(:,2));
xlim([-0.05,1.05]);
ylim([-0.05,1.05]);
% The 3-sigma rule:
%   About 68.27% of the values lie within 1 standard deviation of the mean.
%   Similarly, about 95.45% of the values lie within 2 standard deviations
%   of the mean. Nearly all (99.73%) of the values lie within 3 standard
%   deviations of the mean.
%std(centroids)
% ^^ consider here that the max. distance from the center is 
%      sqrt(0.5^2 + 0.5^2) = 0.7071
%    and std(centroids) is 0.1494, which means that 99 percent of the
%    points are in the area [0.5 +- 3*0.1494 ; 0.5 +- 3*0.1464], i.e.
%    between the points [-0.0518  0.0608]^T and [0.9482 0.9392]^T, which
%    is perfectly fine. => that are reasonable variances

%% Run statistical tests on the Radii
alpha_value=0.05;

% truncated distribution ...
%test_angles = [radius_centroids_normalized.*sign(angles_centroids); -radius_centroids_normalized.*sign(angles_centroids)];
test_angles = radius_centroids_normalized.*sign(angles_centroids);

fprintf('Testing radii vs Normal (II)\n');
[H,p,c,d]=lillietest(test_angles,alpha_value);
fprintf('  test=%s (vs %s) H=%d p=%f\n','Lilliefors','normal',H,p);
[H,p,c,d]=jbtest(test_angles,alpha_value);
fprintf('  test=%s (vs %s) H=%d p=%f\n','Jarque-Bera','normal',H,p);
[H,p]=swtest(test_angles,alpha_value);
fprintf('  test=%s (vs %s) H=%d p=%f\n','Shapiro-Wilk','normal',H,p);
p=spiegelhaltertest(test_angles);
H = (p < alpha_value);
fprintf('  test=%s (vs %s) H=%d p=%f\n','Spiegelhalter','normal',H,p);

%% Run statistical tests on the Angles
alpha_value=0.05;

% use a Pearson Chi-Square test to test whether or not the angles are
% sampled from a uniform distribution
poisscdf=@(x) cdf('poiss',x);
tcdf=@(x) cdf('t',x,1);
cauchycdf=@(x) (0.5 + (atan(x)/pi));

fprintf('Testing angles\n');
[H,p,stats]=chi2gof(angles_centroids,'cdf',@unifcdf,'alpha',alpha_value); % uniform distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Chi-Square','uniform',H,p);
[H,p,stats]=chi2gof(angles_centroids,'alpha',alpha_value); % normal distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Chi-Square','normal',H,p);
[H,p,stats]=chi2gof(angles_centroids,'cdf',@expcdf,'alpha',alpha_value); % exponential distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Chi-Square','exponential',H,p);
[H,p,stats]=chi2gof(angles_centroids,'cdf',poisscdf,'alpha',alpha_value); % exponential distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Chi-Square','poisson',H,p);
[H,p,stats]=chi2gof(angles_centroids,'cdf',tcdf,'alpha',alpha_value); % exponential distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Chi-Square','students t',H,p);
[H,p,stats]=chi2gof(angles_centroids,'cdf',cauchycdf,'alpha',alpha_value); % Cauchy distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Chi-Square','cauchy',H,p);
[H,p,stats]=lillietest(angles_centroids,alpha_value,'norm',0.001); % normal distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Lilliefors','normal',H,p);
[H,p,stats]=lillietest(angles_centroids,alpha_value,'exp',0.001); % exponential distribution
fprintf('  test=%s (vs %s) H=%d p=%f\n','Lilliefors','exponential',H,p);

%% Run tests against circular distributions
if isempty(which('circ_rtest'))
  fprintf('Missing library: Testing circular distributions requires the circstat package.\n'); 
else  
  fprintf('Testing angles vs circular\n');
  figure('name','centroid angles');
  circ_plot(angles_centroids);
  [p z]=circ_rtest(angles_centroids);
  H = p < 0.05;
  fprintf('  test=%s (vs %s) H=%d p=%f\n','R-test','uniform',H,p);
  [p z]=circ_raotest(angles_centroids);
  H = p < 0.05;
  fprintf('  test=%s (vs %s) H=%d p=%f\n','Rao-test','uniform',H,p);
end

%% Q-Q Plots
figure('name','Q-Q-Plots');

% Angles Q-Q-Plot
subplot(1,3,1);
qqplot(angles_centroids,(rand(numel(angles_centroids)*10,1)-0.5)*pi*2)
xlab = 'Standard Uniform Quantiles';
ylab = 'Centroid Angle Quantiles'; %'Quantiles of Input Sample';
tlab = 'Q-Q Plot of Sample Data versus Standard Uniform';
title ('Angles');
set(gca,'XTickLabel',{});
set(gca,'YTickLabel',{});
axis('square');

% Radius Q-Q-Plot
subplot(1,3,2);
qqplot(radius_centroids_normalized.*sign(angles_centroids))
xlab = 'Standard Normal Quantiles';
ylab = 'Transformed Radius Quantiles';
tlab = 'Q-Q Plot of Sample Data versus Standard Normal'; %'Quantiles of Input Sample';
title ('Radii');
set(gca,'XTickLabel',{});
set(gca,'YTickLabel',{});
axis('square');

% Truncated Radius Normal Q-Q-Plot
if isempty(which('randraw'))
  fprintf('Missing library: Q-Q plotting truncated normal radii requires the circstat package.\n'); 
else
  subplot(1,3,3);
  qqplot(radius_centroids_normalized,randraw('normaltrunc',[0,Inf,0,max(radius_centroids_normalized)],numel(radius_centroids_normalized)*20))
  xlab = 'Half-Gaussian Quantiles';
  ylab = 'Radius Quantiles';
  tlab = 'Q-Q Plot of Sample Data versus Standard Normal'; %'Quantiles of Input Sample';
  xlim([-0.1*0.8 0.8])
  ylim([-0.1*2.5 2.5])
  title ('Truncated Radii');
  set(gca,'XTickLabel',{});
  set(gca,'YTickLabel',{});
  axis('square');
end

%% Let's check the probability plot correlation coefficient (PPCC)
N=numel(angles_centroids)*100;
B=100; % number of quantiles
pvec = 100*((1:B) - 0.5) ./ B;

% calculate the percentiles of our data
angles_centroids_percentiles=prctile(angles_centroids(1:1000),pvec);
radius_centroids_percentiles=prctile(radius_centroids,pvec);
radius_centroids_normalized_percentiles=prctile(radius_centroids_normalized,pvec);
%transformed_radius_centroids_percentiles=prctile(radius_centroids.*sign(angles_centroids),pvec);
transformed_radius_centroids_percentiles=prctile(radius_centroids(1:1000).*sign(angles_centroids(1:1000)),pvec); % limit this to 1000 samples so that we can use the tabulated critical value
% calculate the percentiles samples from the target distributions
uniform_percentiles=prctile(rand(N,1),pvec);
normal_percentiles=prctile(randraw('norm', [], N), pvec);
lognormal_percentiles=prctile(randraw('lognorm', [], N), pvec);
weibull_percentiles=prctile(randraw('weibull', [0 1 1], N), pvec);
exp_percentiles=prctile(randraw('exp', [1], N), pvec);
ushaped_percentiles=prctile(randraw('u', [], N), pvec);
tri_precentiles=prctile(randraw('tri',[0 1 2], N), pvec);
cauchy_percentiles=prctile(randraw('cauchy', [0 2], N), pvec);

% also see: "Normal Probability Plots and Tests for Normality" by Rand

% see http://www.itl.nist.gov/div898/handbook/eda/section3/eda3676.htm
% (i.e. Sec. 1.3.6.7.6. "Critical Values of the Normal PPCC Distribution")
%
% also see http://engineering.tufts.edu/cee/people/vogel/publications/lowFlowFrequency.pdf
% for the critical values for the uniform distribution
% "LOW-FLOW FREQUENCY ANALYSIS USING PROBABILITY-PLOT CORRELATION
%  COEFFICIENTS" By Richard M. Vogel and Charles N. Kroll, Journal of Water
%  Resources Planning and Management, Vol. 115, No.3, May, 1989

fprintf('\n');

% calculate the probability plot correlation coefficients
cc=corrcoef(angles_centroids_percentiles,uniform_percentiles);
fprintf('Correlation coefficient of Angles vs Uniform:     %f > %f\n',cc(2),0.888);
% Critical values (http://www.itl.nist.gov/div898/handbook/eda/section3/eda3676.htm)
%   N           \alpha=0.01   \alpha=0.05
%   1000        1.32        0.888
fprintf('--\n');
cc=corrcoef(angles_centroids_percentiles,normal_percentiles);
fprintf('Correlation coefficient of Angles vs Normal:      %f < %f\n',cc(2),0.888);
cc=corrcoef(angles_centroids_percentiles,lognormal_percentiles);
fprintf('Correlation coefficient of Angles vs LogNormal:   %f\n',cc(2));
cc=corrcoef(angles_centroids_percentiles,weibull_percentiles);
fprintf('Correlation coefficient of Angles vs Weibull:     %f\n',cc(2));
cc=corrcoef(angles_centroids_percentiles,exp_percentiles);
fprintf('Correlation coefficient of Angles vs Exponential: %f\n',cc(2));
cc=corrcoef(angles_centroids_percentiles,ushaped_percentiles);
fprintf('Correlation coefficient of Angles vs U:           %f\n',cc(2));
cc=corrcoef(angles_centroids_percentiles,tri_precentiles); % probably this value is so high, because corrcoef is a measure for linear correlation (im Wesentlich hier Korrelation zwischen zwei Linien => hoher Wert kann erwartet werden)
fprintf('Correlation coefficient of Angles vs Triangular:  %f\n',cc(2));
cc=corrcoef(angles_centroids_percentiles,cauchy_percentiles);
fprintf('Correlation coefficient of Angles vs Cauchy:      %f\n',cc(2));

fprintf('\n');

% calculate the probability plot correlation coefficients
cc=corrcoef(transformed_radius_centroids_percentiles,normal_percentiles);
fprintf('Correlation coefficient of Transformed Radii vs Normal:      %f > %f\n',cc(2),0.9984); 
% Critical values (http://www.itl.nist.gov/div898/handbook/eda/section3/eda3676.htm)
%   N           \alpha=0.01   \alpha=0.05
%   1000        0.9979        0.9984
fprintf('--\n');
cc=corrcoef(transformed_radius_centroids_percentiles,lognormal_percentiles);
fprintf('Correlation coefficient of Transformed Radii vs LogNormal:   %f\n',cc(2));
cc=corrcoef(transformed_radius_centroids_percentiles,uniform_percentiles);
fprintf('Correlation coefficient of Transformed Radii vs Uniform:     %f < %f\n',cc(2),0.9984);
cc=corrcoef(transformed_radius_centroids_percentiles,weibull_percentiles);
fprintf('Correlation coefficient of Transformed Radii vs Weibull:     %f\n',cc(2));
cc=corrcoef(transformed_radius_centroids_percentiles,exp_percentiles);
fprintf('Correlation coefficient of Transformed Radii vs Exponential: %f\n',cc(2));
cc=corrcoef(transformed_radius_centroids_percentiles,ushaped_percentiles);
fprintf('Correlation coefficient of Transformed Radii vs U:           %f\n',cc(2));
cc=corrcoef(transformed_radius_centroids_percentiles,tri_precentiles);
fprintf('Correlation coefficient of Transformed Radii vs Triangular:  %f\n',cc(2));
cc=corrcoef(transformed_radius_centroids_percentiles,cauchy_percentiles);
fprintf('Correlation coefficient of Transformed Radii vs Cauchy:      %f\n',cc(2));

% %% For a given distribution with parameters: Calculate the maximum plot correlation
% parameter_values=[0.001:0.01:1];
% %parameter_values=[1:1:100];
% cc_values=zeros(numel(parameter_values),1);
% %target_quantiles=transformed_radius_centroids_percentiles;
% target_quantiles=transformed_radius_centroids_percentiles;
% for i=1:numel(parameter_values)
%   %test_percentiles=prctile(randraw('weibull', [0 parameter_values(i) 1], N), pvec);%prctile(randraw('exp', [parameter_values(i)], N), pvec);
%   %test_percentiles=prctile(randraw('cauchy', [0 parameter_values(i)], N), pvec);%prctile(randraw('exp', [parameter_values(i)], N), pvec);
%   test_percentiles=prctile(randraw('cauchy', [0 parameter_values(i)], N), pvec);%prctile(randraw('exp', [parameter_values(i)], N), pvec);
%   cc=corrcoef(target_quantiles,test_percentiles);
%   cc_values(i) = cc(2);
% end
% figure('name',['PPCC (max=' num2str(max(cc_values)) ')']), plot(cc_values);

% %% Analyse the tail in the truncated normal distribution plot
% [sorted_radius_centroids_normalized I]=sort(radius_centroids_normalized);
% threshold=0.51;
% thresholded_radius_centroids_normalized=sorted_radius_centroids_normalized(sorted_radius_centroids_normalized < threshold);
% numel(sorted_radius_centroids_normalized)-numel(thresholded_radius_centroids_normalized)
% figure, qqplot(thresholded_radius_centroids_normalized,randraw('normaltrunc',[0,Inf,0,1],numel(radius_centroids_normalized)*20))
% figure, qqplot(thresholded_radius_centroids_normalized,randraw('normaltrunc',[0,2.5,0,1],numel(radius_centroids_normalized)*20))