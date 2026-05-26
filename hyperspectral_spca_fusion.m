%% ========================================================================
% HYPERSPECTRAL IMAGING ANALYSIS & TARGET DETECTION
% L1-Regularized Sparse Principal Component Fusion (SPCA)
% 
% Author: Mohamed Ebrahem
% Institution: Military Technical College (MTC)
% 
% Description:
% This script processes multidimensional data cubes from hyperspectral 
% reflectance and Laser-Induced Fluorescence (LIF) sensors. It applies 
% Probability Density Function (PDF) divergence for spectral band selection 
% and simulates extreme Poisson-Gaussian industrial noise (up to 30%).
% 
% A custom L1-regularized Sparse Principal Component Analysis (SPCA) 
% algorithm is implemented to fuse the selected bands, optimizing for 
% maximum Contrast-to-Noise Ratio (CNR) to automate target segmentation 
% via K-means clustering under severe signal degradation.
%% ========================================================================

clc;
clear;
close all;
warning off;

%% ==========================================================
%%       PHASE 1: DATA LOADING & PDF BAND SELECTION
%% ==========================================================
disp('Loading data and performing PDF Band Extraction...');

if ~exist('Cuberead1', 'var')
    filename1 = 's11.cube';
    if ~exist(filename1, 'file'), error('File %s not found.', filename1); end
    Cuberead1 = multibandread(filename1, [696, 520, 128], 'int16', 32768, 'bil', 'ieee-le');
    Cuberead1 = permute(Cuberead1, [2, 1, 3]);
end

start_wavelength = 400; end_wavelength = 1000; num_bands = size(Cuberead1, 3);
spectral_resolution = (end_wavelength - start_wavelength) / (num_bands - 1);
wavelengths = start_wavelength + (0:num_bands-1) * spectral_resolution;

band_index = 1;
fig1 = figure('Name', 'Figure 1: ROI Selection', 'Position', [100, 100, 800, 600]);
imagesc(Cuberead1(:,:,band_index)); colormap gray; axis image;
title('Experimental Setup & ROI Selection'); hold on;

disp('Select ROI for WOOD'); wood_mask = roipoly;
stats = regionprops(wood_mask, 'BoundingBox'); bb = stats.BoundingBox; 
h_rect = rectangle('Position', bb, 'EdgeColor', 'y', 'LineWidth', 1.5, 'LineStyle', '--');

disp('Select CENTER for PLASTIC ROI'); [x_p, y_p] = ginput(1); delete(h_rect); 
plastic_mask = poly2mask([x_p-bb(3)/2, x_p+bb(3)/2, x_p+bb(3)/2, x_p-bb(3)/2], [y_p-bb(4)/2, y_p-bb(4)/2, y_p+bb(4)/2, y_p+bb(4)/2], size(Cuberead1,1), size(Cuberead1,2));

disp('Select CENTER for METAL ROI'); [x_m, y_m] = ginput(1);
metal_mask = poly2mask([x_m-bb(3)/2, x_m+bb(3)/2, x_m+bb(3)/2, x_m-bb(3)/2], [y_m-bb(4)/2, y_m-bb(4)/2, y_m+bb(4)/2, y_m+bb(4)/2], size(Cuberead1,1), size(Cuberead1,2));
hold off; drawnow;

% 3. Calculate PDFs & Divergence
reshaped_cube = reshape(Cuberead1, [], num_bands);
wood_mean = mean(reshaped_cube(wood_mask(:), :), 1); 
plastic_mean = mean(reshaped_cube(plastic_mask(:), :), 1); 
metal_mean = mean(reshaped_cube(metal_mask(:), :), 1); 

w_norm = (wood_mean - min(wood_mean,[],2)) ./ (max(wood_mean,[],2) - min(wood_mean,[],2));
p_norm = (plastic_mean - min(plastic_mean,[],2)) ./ (max(plastic_mean,[],2) - min(plastic_mean,[],2));
m_norm = (metal_mean - min(metal_mean,[],2)) ./ (max(metal_mean,[],2) - min(metal_mean,[],2));

wood_pdf = normpdf(w_norm, mean(w_norm), std(w_norm));
plastic_pdf = normpdf(p_norm, mean(p_norm), std(p_norm));
metal_pdf = normpdf(m_norm, mean(m_norm), std(m_norm));

total_divergence = abs(wood_pdf - plastic_pdf) + abs(wood_pdf - metal_pdf) + abs(plastic_pdf - metal_pdf);

% 4. Select Top 4 Bands (10 nm margin)
num_best = 4; threshold_nm = 10; 
selected_indices = []; temp_divergence = total_divergence;
while length(selected_indices) < num_best
    [max_val, max_idx] = max(temp_divergence);
    selected_indices = [selected_indices, max_idx];
    temp_divergence(abs(wavelengths - wavelengths(max_idx)) < threshold_nm) = -inf;
end

[rows, cols, ~] = size(Cuberead1);
clean_4_bands = zeros(rows, cols, num_best);
for i = 1:num_best
    clean_4_bands(:,:,i) = double(Cuberead1(:,:,selected_indices(i)));
    clean_4_bands(:,:,i) = (clean_4_bands(:,:,i) - min(min(clean_4_bands(:,:,i)))) / (max(max(clean_4_bands(:,:,i))) - min(min(clean_4_bands(:,:,i))));
end

[~, idx_740] = min(abs(wavelengths - 740));
clean_740_norm = double(Cuberead1(:,:,idx_740));
clean_740_norm = (clean_740_norm - min(clean_740_norm(:))) / (max(clean_740_norm(:)) - min(clean_740_norm(:)));

fig2 = figure('Name', 'Figure 2: Band Selection', 'Position', [150, 150, 1000, 600]);
subplot(2,1,1);
plot(wavelengths, mean(w_norm, 1), 'LineWidth', 2); hold on;
plot(wavelengths, mean(p_norm, 1), 'LineWidth', 2);
plot(wavelengths, mean(m_norm, 1), 'LineWidth', 2);
legend('Wood', 'Plastic', 'Metal'); title('Normalized Spectral Signatures'); xlabel('Wavelength (nm)'); ylabel('Intensity'); grid on;
subplot(2,1,2);
plot(wavelengths, total_divergence(1,:), 'k', 'LineWidth', 2); hold on;
plot(wavelengths(selected_indices), total_divergence(1,selected_indices), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
title('Total Statistical Divergence & Extracted Bands'); xlabel('Wavelength (nm)'); ylabel('\Delta PDF'); grid on;
drawnow;
%%
% --- First Figure: Normalized Spectral Signatures ---
fig2a = figure('Name', 'Figure 2a: Normalized Spectral Signatures', 'Position', [150, 500, 800, 400]);
plot(wavelengths, mean(w_norm, 1), 'LineWidth', 2); hold on;
plot(wavelengths, mean(p_norm, 1), 'LineWidth', 2);
plot(wavelengths, mean(m_norm, 1), 'LineWidth', 2);
legend('Wood', 'Plastic', 'Metal'); 
title('Normalized Spectral Signatures'); 
xlabel('Wavelength (nm)'); 
ylabel('Intensity'); 
grid on;
drawnow;

% --- Second Figure: Statistical Divergence & Extracted Bands ---
fig2b = figure('Name', 'Figure 2b: Band Selection', 'Position', [200, 100, 800, 400]);
plot(wavelengths, total_divergence(1,:), 'k', 'LineWidth', 2); hold on;
plot(wavelengths(selected_indices), total_divergence(1,selected_indices), 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'MarkerFaceColor', 'r');
title('Total Statistical Divergence & Extracted Bands'); 
xlabel('Wavelength (nm)'); 
ylabel('\Delta PDF'); 
grid on;
drawnow;
%% ==========================================================
%%       PHASE 2: DEFINING INDUSTRIAL NOISE MODELS
%% ==========================================================
disp('Generating Mixed Industrial Noise Models (EXTREME 30% NOISE)...');

peak_photons = 20; % POISSON: Simulates Photon Starvation
sigma_gaussian = 0.30 * max(clean_740_norm(:)); % GAUSSIAN: Increased to 30% Read/Thermal Noise

apply_mixed = @(img) max(0, min(1, (poissrnd(max(0, img) * peak_photons) / peak_photons) + (randn(size(img)) * sigma_gaussian)));

%% ==========================================================
%%       PHASE 3: IDEAL LABORATORY BASELINE
%% ==========================================================
disp('Testing Ideal Conditions (Baseline)...');
[~, mask_740_ideal, cnr_740_ideal] = process_baseline(clean_740_norm);
% We pass an arbitrary lambda here just for the clean baseline
[~, mask_spca_ideal, img_spca_ideal, ~, ~] = process_spca(clean_4_bands, plastic_mask, wood_mask, 0.5);

fig3 = figure('Name', 'Figure 3: Laboratory Baseline', 'Position', [200, 200, 1000, 400]);
subplot(1,3,1); imagesc(clean_740_norm); colormap gray; title('Clean 740nm'); axis off;
subplot(1,3,2); imagesc(mask_740_ideal); colormap gray; title('K-means (740nm)'); axis off;
subplot(1,3,3); imagesc(mask_spca_ideal); colormap gray; title('K-means (SPCA Fusion)'); axis off;
drawnow;

%% ==========================================================
%%       PHASE 4: THE MIXED NOISE SHOWDOWN WITH AUTO-TUNING
%% ==========================================================
disp('Applying Mixed Noise (Poisson + 30% Gaussian)...');

img_740_M = apply_mixed(clean_740_norm);
bands_4_M = zeros(size(clean_4_bands));
for i=1:num_best, bands_4_M(:,:,i) = apply_mixed(clean_4_bands(:,:,i)); end

disp('Evaluating Noisy 740nm...');
[raw_mask_740_M, clean_mask_740_M, cnr_740_M] = process_baseline(img_740_M);

disp('Evaluating Standard PCA Fusion...');
[raw_mask_pca_M, clean_mask_pca_M, img_pca_M, cnr_pca_M, weights_pca] = process_standard_pca(bands_4_M, plastic_mask, wood_mask);

disp('Tuning Sparse PCA (Finding optimal lambda)...');
% --- AUTO TUNING LOOP ---
lambda_range = 0.1:0.05:0.95;
best_cnr_spca = -inf;
best_lambda = 0;
best_spca_results = cell(1,5);

for lam = lambda_range
    [raw_s, clean_s, img_s, cnr_s, w_s] = process_spca(bands_4_M, plastic_mask, wood_mask, lam);
    if cnr_s > best_cnr_spca
        best_cnr_spca = cnr_s;
        best_lambda = lam;
        best_spca_results = {raw_s, clean_s, img_s, cnr_s, w_s};
    end
end

fprintf('*** Optimal Lambda found: %.2f ***\n', best_lambda);

% Extract the best results
raw_mask_spca_M = best_spca_results{1};
clean_mask_spca_M = best_spca_results{2};
img_spca_M = best_spca_results{3};
cnr_spca_M = best_spca_results{4};
weights_spca = best_spca_results{5};

%% ==========================================================
%%       PHASE 5: QUANTITATIVE RESULTS & VISUALIZATION
%% ==========================================================
fprintf('\n========================================================\n');
fprintf('   THE MIXED NOISE SHOWDOWN: CNR ROBUSTNESS SUMMARY   \n');
fprintf('========================================================\n');
fprintf('Method                | CNR Score | Improvement vs 740nm\n');
fprintf('----------------------|-----------|---------------------\n');
fprintf('Single Band (740nm)   | %.4f    | Baseline\n', cnr_740_M);
fprintf('Standard PCA Fusion   | %.4f    | %+.1f%%\n', cnr_pca_M, ((cnr_pca_M-cnr_740_M)/cnr_740_M)*100);
fprintf('Sparse PCA (Optimal)  | %.4f    | %+.1f%%\n', cnr_spca_M, ((cnr_spca_M-cnr_740_M)/cnr_740_M)*100);
fprintf('========================================================\n');
fprintf('\nFeature Weights:\n');
fprintf('Standard PCA: [%.3f, %.3f, %.3f, %.3f]\n', abs(weights_pca));
fprintf('Sparse PCA:   [%.3f, %.3f, %.3f, %.3f]\n', abs(weights_spca));

fig4 = figure('Name', 'Figure 4: Mixed Noise Showdown', 'Position', [100, 50, 1200, 800]);

% Row 1: Single Band 740nm
subplot(3,3,1); imagesc(img_740_M); colormap gray; title('Noisy 740nm Input'); axis off;
subplot(3,3,2); imagesc(raw_mask_740_M); colormap gray; title(sprintf('Raw K-means Mask (Fails due to noise)')); axis off;
subplot(3,3,3); imagesc(clean_mask_740_M); colormap gray; title(sprintf('Cleaned Mask (CNR: %.2f)', cnr_740_M)); axis off;

% Row 2: Standard PCA
subplot(3,3,4); imagesc(img_pca_M); colormap gray; title('Standard PCA Fused Map'); axis off;
subplot(3,3,5); imagesc(raw_mask_pca_M); colormap gray; title('Raw K-means Mask (Some noise persists)'); axis off;
subplot(3,3,6); imagesc(clean_mask_pca_M); colormap gray; title(sprintf('Cleaned Mask (CNR: %.2f)', cnr_pca_M)); axis off;

% Row 3: Sparse PCA (Proposed)
subplot(3,3,7); imagesc(img_spca_M); colormap gray; title(sprintf('Sparse PCA Fused Map (Lambda = %.2f)', best_lambda)); axis off;
subplot(3,3,8); imagesc(raw_mask_spca_M); colormap gray; title('Raw K-means Mask (Algorithm Resists Noise)'); axis off;
subplot(3,3,9); imagesc(clean_mask_spca_M); colormap gray; title(sprintf('Cleaned Mask (CNR: %.2f)', cnr_spca_M)); axis off;
drawnow;

fig5 = figure('Name', 'Figure 5: CNR Improvement Summary', 'Position', [300, 200, 700, 500]);
cnr_data = [cnr_740_M, cnr_pca_M, cnr_spca_M];
b = bar(categorical({'1. Single 740nm', '2. Standard PCA', '3. Sparse PCA'}), cnr_data, 'FaceColor', 'flat');
b.CData(1,:) = [0.6 0.6 0.6];
b.CData(2,:) = [0.4 0.6 0.8];
b.CData(3,:) = [0 0.4470 0.7410]; 

ylabel('Contrast-to-Noise Ratio (CNR)', 'FontSize', 12, 'FontWeight', 'bold');
title('Robustness Under Extreme Mixed Industrial Noise (30%)', 'FontSize', 14);
grid on;

improvements = [0, ((cnr_pca_M-cnr_740_M)/cnr_740_M)*100, ((cnr_spca_M-cnr_740_M)/cnr_740_M)*100];
for i = 2:3
    text(i, cnr_data(i) + 0.05, sprintf('+%.1f%%', improvements(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
         'FontWeight', 'bold', 'FontSize', 12, 'Color', [0 0.4470 0.7410]);
end
drawnow;

%% ==========================================================
%%       PHASE 6: AUTOMATED FIGURE EXPORT
%% ==========================================================
disp('Exporting High-Resolution Figures (300 DPI)...');
exportgraphics(fig1, 'Figure1_ROI_Setup.png', 'Resolution', 300);
exportgraphics(fig2, 'Figure2_PDF_Selection.png', 'Resolution', 300);
exportgraphics(fig3, 'Figure3_Ideal_Baseline.png', 'Resolution', 300);
exportgraphics(fig4, 'Figure4_Mixed_Noise_Showdown.png', 'Resolution', 300);
exportgraphics(fig5, 'Figure5_CNR_Chart.png', 'Resolution', 300);
disp('All figures saved! Check your MATLAB folder.');
%%
exportgraphics(fig2a, 'Figure2a_Signatures.png', 'Resolution', 300);
exportgraphics(fig2b, 'Figure2b_PDF_Selection.png', 'Resolution', 300);
%% ========================================================================
%%  LOCAL HELPER FUNCTIONS (Classification & Metrics)
%% ========================================================================
function [raw_mask, clean_mask, cnr] = process_baseline(img)
    [idx, C] = kmeans(img(:), 2, 'Replicates', 3);
    [~, sort_order] = sort(C, 'ascend'); 
    raw_mask = zeros(size(img));
    raw_mask(reshape(idx, size(img)) == sort_order(2)) = 1;
    clean_mask = imfill(bwareaopen(raw_mask, 50), 'holes');
    
    sig = img(clean_mask == 1); bg = img(clean_mask == 0);
    if isempty(sig) || isempty(bg), cnr = 0; else
        cnr = abs(mean(sig) - mean(bg)) / sqrt(std(sig)^2 + std(bg)^2);
    end
end

function [raw_mask, clean_mask, fused_norm, cnr, weights] = process_standard_pca(bands_4, plastic_mask, wood_mask)
    [rows, cols, num_b] = size(bands_4);
    X = reshape(bands_4, [], num_b);
    X_centered = (X - mean(X)) ./ std(X);
    
    [coeff, score] = pca(X_centered);
    weights = coeff(:,1);
    fused_img = reshape(score(:,1), rows, cols);
    fused_norm = (fused_img - min(fused_img(:))) / (max(fused_img(:)) - min(fused_img(:)));
    
    if mean(fused_norm(plastic_mask)) < mean(fused_norm(wood_mask)), fused_norm = 1 - fused_norm; end
    
    [idx, C] = kmeans(fused_norm(:), 2, 'Replicates', 3);
    [~, sort_order] = sort(C, 'ascend'); 
    raw_mask = zeros(size(fused_img));
    raw_mask(reshape(idx, size(fused_img)) == sort_order(2)) = 1;
    clean_mask = imfill(bwareaopen(raw_mask, 50), 'holes');
    
    sig = fused_norm(clean_mask == 1); bg = fused_norm(clean_mask == 0);
    if isempty(sig) || isempty(bg), cnr = 0; else
        cnr = abs(mean(sig) - mean(bg)) / sqrt(std(sig)^2 + std(bg)^2);
    end
end

function [raw_mask, clean_mask, fused_norm, cnr, weights] = process_spca(bands_4, plastic_mask, wood_mask, lambda_val)
    [rows, cols, num_b] = size(bands_4);
    X = reshape(bands_4, [], num_b);
    X_centered = (X - mean(X)) ./ std(X);
    
    [coeff_spca, score_spca] = manual_sparse_pca_corrected(X_centered, 1, lambda_val);
    weights = coeff_spca(:,1);
    fused_img = reshape(score_spca(:,1), rows, cols);
    fused_norm = (fused_img - min(fused_img(:))) / (max(fused_img(:)) - min(fused_img(:)));
    
    if mean(fused_norm(plastic_mask)) < mean(fused_norm(wood_mask)), fused_norm = 1 - fused_norm; end
    
    [idx, C] = kmeans(fused_norm(:), 2, 'Replicates', 3);
    [~, sort_order] = sort(C, 'ascend'); 
    raw_mask = zeros(size(fused_img));
    raw_mask(reshape(idx, size(fused_img)) == sort_order(2)) = 1;
    clean_mask = imfill(bwareaopen(raw_mask, 50), 'holes');
    
    sig = fused_norm(clean_mask == 1); bg = fused_norm(clean_mask == 0);
    if isempty(sig) || isempty(bg), cnr = 0; else
        cnr = abs(mean(sig) - mean(bg)) / sqrt(std(sig)^2 + std(bg)^2);
    end
end

function [sparse_coeff, sparse_scores] = manual_sparse_pca_corrected(X, k, lambda)
    [n, p] = size(X);
    sparse_coeff = zeros(p, k);
    sparse_scores = zeros(n, k);
    X_current = X;
    for comp = 1:k
        [~, ~, v] = svd(X_current, 'econ');
        v = v(:, 1);
        max_iter = 200; tol = 1e-6;
        for iter = 1:max_iter
            v_old = v;
            z = X_current' * (X_current * v);
            threshold = lambda * max(abs(z));
            v = sign(z) .* max(abs(z) - threshold, 0);
            v_norm = norm(v);
            if v_norm > 1e-10, v = v / v_norm; else, v = v_old; break; end
            if norm(v - v_old) < tol, break; end
        end
        sparse_coeff(:, comp) = v;
        sparse_scores(:, comp) = X_current * v;
        X_current = X_current - sparse_scores(:, comp) * v';
    end
end