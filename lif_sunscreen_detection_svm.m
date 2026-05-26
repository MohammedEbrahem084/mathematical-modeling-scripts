%% ========================================================================
% LASER-INDUCED FLUORESCENCE (LIF) HYPERSPECTRAL IMAGING ANALYSIS
% End-to-End Machine Learning Pipeline for Anomaly Detection
% 
% Author: Mohamed Ebrahem
% Institution: Military Technical College (MTC)
% 
% Description:
% This script implements a comprehensive machine learning pipeline for 
% detecting trace contamination (sunscreen residue) in food-grade sea salt 
% using LIF Hyperspectral Imaging. 
%
% It features rigorous mathematical data preprocessing (baseline correction, 
% area normalization, Savitzky-Golay smoothing), Hyperparameter Optimization, 
% L1-Regularized Sparse Principal Component Analysis (SPCA) for feature 
% extraction, and Support Vector Machine (SVM) classification with k-fold 
% cross-validation and ground-truth performance evaluation.
%% ========================================================================

clc;
clear;
close all;
warning off;

%% ========================================================================
%  SECTION 1: HYPERSPECTRAL DATA LOADING
% ========================================================================
% Load the hyperspectral cube acquired from SOC710 camera
% Cube dimensions: 696 x 520 pixels x 128 spectral bands
% Spectral range: 400-1000 nm
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 1: LOADING HYPERSPECTRAL DATA\n');
fprintf('========================================\n');

if ~exist('Cuberead1', 'var')
    filename1 = 'S12.cube';
    if ~exist(filename1, 'file')
        error('File %s not found. Please check the file path.', filename1);
    end
    
    fprintf('Loading hyperspectral cube: %s\n', filename1);
    % Read cube with BIL (Band Interleaved by Line) format
    Cuberead1 = multibandread(filename1, [696, 520, 128], 'int16', 32768, 'bil', 'ieee-le');
    
    % Permute dimensions to [rows, columns, bands] for easier processing
    Cuberead1 = permute(Cuberead1, [2, 1, 3]);
    fprintf('Cube loaded successfully: %d x %d x %d\n', size(Cuberead1,1), size(Cuberead1,2), size(Cuberead1,3));
end
%% ========================================================================
%  LOAD SECOND HYPERSPECTRAL CUBE (FOR CLEAN SALT ROI)
% ========================================================================

fprintf('\n========================================\n');
fprintf('LOADING SECOND CUBE FOR CLEAN SALT ROI\n');
fprintf('========================================\n');

if ~exist('Cuberead2', 'var')
    filename2 = 'Salt1.cube';   % <=== CHANGE THE NAME OF YOUR NEW CUBE HERE
    if ~exist(filename2, 'file')
        error('File %s not found. Please check the file path.', filename2);
    end
    
    fprintf('Loading second hyperspectral cube: %s\n', filename2);
    Cuberead2 = multibandread(filename2, [696, 520, 128], ...
        'int16', 32768, 'bil', 'ieee-le');

    Cuberead2 = permute(Cuberead2, [2, 1, 3]);
    fprintf('Second cube loaded successfully: %d x %d x %d\n', ...
        size(Cuberead2,1), size(Cuberead2,2), size(Cuberead2,3));
end

%% ========================================================================
%  SECTION 1b: GROUND TRUTH MASK CREATION (OPTIONAL)
% ========================================================================
% Create ground truth masks if they don't exist
% ========================================================================

if ~exist('GT_Masks.mat', 'file')
    fprintf('\n========================================\n');
    fprintf('SECTION 1b: GROUND TRUTH MASK CREATION\n');
    fprintf('========================================\n');
    
    % Assumes Cuberead1 (the hyperspectral cube) is already loaded
    % We'll use a suitable band for displaying the image during annotation
    img2show = Cuberead1(:,:,23); % Choose the display band as needed
    % Show with color map for initial visualization
    figure;
    imagesc(img2show);
    colormap('gray');
    colorbar;
    title('Select regions on pseudo-color map (Band 23)');

    num_ROIs = input('How many regions (masks) do you want to annotate? '); % Enter the number of masks
    Masks = false([size(img2show), num_ROIs]); % Array to hold each mask

for i = 1:num_ROIs
    figure;
    imshow(img2show, []); % This is correct—imshow wants image data
    colormap('gray'); % Apply color map if you want during ROI selection
    colorbar;
    title(['Select region number ', num2str(i), '. Draw ROI then double-click or press Enter']);
    mask = roipoly;
    Masks(:,:,i) = mask;
    figure;
    imshow(img2show, []);
    colormap('gray');
    colorbar;
    hold on;
    h = imshow(mask);
    set(h, 'AlphaData', 0.3); % Overlay mask on the image
    title(['Mask ', num2str(i), ' overlaid on original image']);
    hold off;
end

    save('GT_Masks.mat', 'Masks');
    disp('All masks have been saved in GT_Masks.mat');
else
    fprintf('Ground truth masks found: GT_Masks.mat\n');
    load('GT_Masks.mat', 'Masks');
end

% Combine all masks into one ground-truth mask (GT)
GT = any(Masks, 3);   % If a pixel belongs to ANY ROI → contaminated

figure; 
imshow(GT, []); title('Ground Truth Mask (1 = contaminated)');
%%
%% ========================================================================
%  SECTION 1b: LOAD AUTOMATED GROUND TRUTH MASK FROM SVM CLASSIFICATION
% ========================================================================
% This replaces manual region selection; uses pre-saved binary mask
% ========================================================================

if exist('GT_mask_binary.mat', 'file')
    fprintf('\n========================================\n');
    fprintf('SECTION 1b: LOADING GROUND TRUTH MASK FROM SVM CLASSIFICATION\n');
    fprintf('========================================\n');
    load('GT_mask_binary.mat', 'GT_binary');
    GT = GT_binary; % GT is now your ground truth mask: 1=contaminated, 0=clean salt
    disp('Ground truth mask loaded from SVM classification output.');
else
    error('GT_mask_binary.mat not found. Please generate and save your automated ground truth mask first.');
end

figure;
imshow(GT, []);
title('Ground Truth Mask ');
colormap(gca, [1 1 1; 0 0 1]);
colorbar('Ticks', [0, 1], 'TickLabels', {'Clean Salt', 'Sunscreen'});

%% ========================================================================
%  SECTION 2: WAVELENGTH AXIS DEFINITION
% ========================================================================
% Define the wavelength corresponding to each spectral band
% SOC710 camera specifications: 400-1000 nm range, 128 bands
% ========================================================================
    
fprintf('\n========================================\n');
fprintf('SECTION 2: WAVELENGTH CALIBRATION\n');
fprintf('========================================\n');

start_wavelength = 400;  % nm
end_wavelength = 1000;   % nm
num_bands = size(Cuberead1, 3);  % 128 bands

% Calculate spectral resolution
spectral_resolution = (end_wavelength - start_wavelength) / (num_bands - 1);
fprintf('Spectral resolution: %.2f nm per band\n', spectral_resolution);

% Generate wavelength vector
wavelengths = start_wavelength + (0:num_bands-1) * spectral_resolution;
fprintf('Wavelength range: %.1f - %.1f nm\n', wavelengths(1), wavelengths(end));

%% ========================================================================
%  SECTION 3: ROI SELECTION ON CLEAN & CONTAMINATED CUBES (FIXED SIZE)
% ========================================================================
% Assumes Cuberead1 (contaminated cube) and Cuberead2 (clean cube) are loaded
% band_index specifies which band to use for visualization (adapt as needed)
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 3: ROI SELECTION (SEPARATE CUBES, FIXED SIZE)\n');
fprintf('========================================\n');

% Choose which band to display for ROI selection
band_index = 23;  % Change as appropriate

% Display CLEAN SALT (Cuberead2)
monoImageClean = Cuberead2(:,:,band_index);
figure('Position', [100, 100, 800, 600], 'Name', 'Clean Salt ROI');
imagesc(monoImageClean);
colormap('gray');
colorbar;
title('Select ROI for CLEAN SALT (use band 24 by default)');
axis image; hold on;

% ROI size selection (apply same size for both cubes)
fprintf('\nDefine FIXED ROI size for both clean and contaminated regions.\n');
roi_width = input('Enter ROI width (pixels): ');
roi_height = input('Enter ROI height (pixels): ');

fprintf('\nClick CENTER of CLEAN SALT region in the CURRENT figure...\n');
[x_clean, y_clean] = ginput(1);

maskClean = poly2mask(...
    [x_clean-roi_width/2, x_clean+roi_width/2, x_clean+roi_width/2, x_clean-roi_width/2], ...
    [y_clean-roi_height/2, y_clean-roi_height/2, y_clean+roi_height/2, y_clean+roi_height/2], ...
    size(monoImageClean,1), size(monoImageClean,2));
boundaryClean = bwboundaries(maskClean);

plot(boundaryClean{1}(:,2), boundaryClean{1}(:,1), 'g', 'LineWidth', 2.5);
legend('Clean Salt ROI');
hold off;

fprintf('CLEAN ROI selected: %d pixels\n', sum(maskClean(:)));

% Display CONTAMINATED SALT (Cuberead1)
monoImageContam = Cuberead1(:,:,band_index);
figure('Position', [950, 100, 800, 600], 'Name', 'Contaminated Salt ROI');
imagesc(monoImageContam);
colormap('gray');
colorbar;
title('Select ROI for SUNSCREEN-CONTAMINATED SALT (use band 24 by default)');
axis image; hold on;

fprintf('\nClick CENTER of SUNSCREEN-CONTAMINATED region in the CURRENT figure...\n');
[x_contam, y_contam] = ginput(1);

maskContam = poly2mask(...
    [x_contam-roi_width/2, x_contam+roi_width/2, x_contam+roi_width/2, x_contam-roi_width/2], ...
    [y_contam-roi_height/2, y_contam-roi_height/2, y_contam+roi_height/2, y_contam+roi_height/2], ...
    size(monoImageContam,1), size(monoImageContam,2));
boundaryContam = bwboundaries(maskContam);

plot(boundaryContam{1}(:,2), boundaryContam{1}(:,1), 'b', 'LineWidth', 2.5);
legend('Contaminated Salt ROI');
hold off;

fprintf('CONTAMINATED ROI selected: %d pixels\n', sum(maskContam(:)));

% Save ROI masks and settings
save('Fixed_ROIs.mat', 'maskClean', 'maskContam', 'roi_width', 'roi_height', ...
    'x_clean', 'y_clean', 'x_contam', 'y_contam');

fprintf('\nROI selection completed successfully and saved to "Fixed_ROIs.mat".\n');
fprintf('Pixel locations - Clean Salt: (%.0f, %.0f) | Contaminated Salt: (%.0f, %.0f)\n', ...
    x_clean, y_clean, x_contam, y_contam);



%% ========================================================================
%  SECTION 4: SPECTRAL SIGNATURE EXTRACTION FROM TWO CUBES
% ========================================================================
fprintf('\n========================================\n');
fprintf('SECTION 4: SPECTRAL SIGNATURE EXTRACTION\n');
fprintf('========================================\n');

% Get the number of bands (assumes Cuberead1 and Cuberead2 are both loaded)
num_bands = size(Cuberead1, 3);

% Reshape cubes to [num_pixels, num_bands] for easy spectral extraction
reshaped_cube1 = reshape(Cuberead1, [], num_bands);  % Contaminated cube
reshaped_cube2 = reshape(Cuberead2, [], num_bands);  % Clean cube

% Extract mean spectra from each mask
clean_salt_signature = mean(reshaped_cube2(maskClean(:), :), 1);              % From clean cube, clean salt ROI
contam_sunscreen_signature = mean(reshaped_cube1(maskContam(:), :), 1);       % From contaminated cube, sunscreen ROI

% Calculate wavelength vector for x-axis (adjust these values as needed)
wavelengths = linspace(400, 1000, num_bands);  % nm, matches camera specs

%% Plot mean spectral signatures with wavelength x-axis
figure;
plot(wavelengths, clean_salt_signature, 'g', 'LineWidth', 2);
hold on;
plot(wavelengths, contam_sunscreen_signature, 'b', 'LineWidth', 2);
xlabel('Wavelength (nm)');
ylabel('Mean Reflectance (a.u.)');
legend('Clean Salt', 'Sunscreen-Contaminated');
title('Mean Spectral Signatures');
grid on;

%% Plot mean spectral signatures (db) with wavelength x-axis
figure;
plot(wavelengths,20*log10(clean_salt_signature), 'g', 'LineWidth', 2);
hold on;
plot(wavelengths,20*log10(contam_sunscreen_signature), 'b', 'LineWidth', 2);
xlabel('Wavelength (nm)');
ylabel('Mean Reflectance (dB)');
legend('Clean Salt', 'Sunscreen-Contaminated');
title('Mean Spectral Signatures in dB');
grid on;

%% ========================================================================
%  SECTION 5: SPECTRAL PREPROCESSING
% ========================================================================
% Fluorescence-specific preprocessing pipeline:
% 1. Remove laser excitation region (400-480 nm) to avoid scatter
% 2. Extract pixel spectra from ROIs
% 3. Apply preprocessing: baseline correction, normalization, smoothing
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 5: SPECTRAL PREPROCESSING\n');
fprintf('========================================\n');

% Step 1: Identify and remove laser excitation bands
laser_band_indices = find(wavelengths >= 400 & wavelengths <= 480);
valid_bands = setdiff(1:num_bands, laser_band_indices);
wavelengths_clean = wavelengths(valid_bands);

fprintf('Laser excitation bands removed: %d-%d nm (%d bands)\n', ...
    wavelengths(laser_band_indices(1)), wavelengths(laser_band_indices(end)), ...
    length(laser_band_indices));
fprintf('Remaining spectral bands: %d (%.0f-%.0f nm)\n', ...
    length(valid_bands), wavelengths_clean(1), wavelengths_clean(end));

% Step 2: Extract pixel spectra from ROIs (FOR TRAINING)
salt_pixels = reshaped_cube2(maskClean(:), valid_bands);          % Clean salt pixels from clean cube
sunscreen_pixels = reshaped_cube1(maskContam(:), valid_bands);    % Sunscreen pixels from contaminated cube

fprintf('\nExtracting pixel spectra from TRAINING ROIs...\n');
fprintf('Salt pixels: %d spectra\n', size(salt_pixels, 1));
fprintf('Sunscreen pixels: %d spectra\n', size(sunscreen_pixels, 1));

% Step 3: Preprocess all pixel spectra
fprintf('\nApplying preprocessing pipeline...\n');
fprintf('- Baseline correction (10th percentile subtraction)\n');
fprintf('- Negative value clipping\n');
fprintf('- Area normalization\n');
fprintf('- Savitzky-Golay smoothing (order=3, window=11)\n');

salt_processed = preprocess_spectra(salt_pixels, wavelengths_clean);
sunscreen_processed = preprocess_spectra(sunscreen_pixels, wavelengths_clean);

fprintf('Preprocessing completed successfully.\n');


%% ========================================================================
%  SECTION 6: CREATE LABELED TRAINING DATASET FROM ROIs
% ========================================================================
% Combine preprocessed spectra into a single feature matrix with labels
% Class 0: Clean salt crystals
% Class 1: Sunscreen-contaminated regions
% USING ROI DATA FOR TRAINING
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 6: TRAINING DATASET PREPARATION (ROI-BASED)\n');
fprintf('========================================\n');

% Concatenate features and create labels FROM ROIs
X_train_roi = [salt_processed; sunscreen_processed];  % Feature matrix (n_samples x n_features)
y_train_roi = [zeros(size(salt_processed, 1), 1);     % Labels: 0 = salt, 1 = sunscreen
     ones(size(sunscreen_processed, 1), 1)];

fprintf('ROI-BASED TRAINING dataset created:\n');
fprintf('  Total samples: %d\n', size(X_train_roi, 1));
fprintf('  Features per sample: %d wavelengths\n', size(X_train_roi, 2));
fprintf('  Class 0 (Salt): %d samples (%.1f%%)\n', ...
    sum(y_train_roi==0), 100*sum(y_train_roi==0)/length(y_train_roi));
fprintf('  Class 1 (Sunscreen): %d samples (%.1f%%)\n', ...
    sum(y_train_roi==1), 100*sum(y_train_roi==1)/length(y_train_roi));

%% ========================================================================
%  SECTION 6b: CREATE GROUND TRUTH VALIDATION DATASET
% ========================================================================
    % Create comprehensive validation dataset from ground truth masks
% This will be used for final performance evaluation
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 6b: GROUND TRUTH VALIDATION DATASET\n');
fprintf('========================================\n');

% Extract pixel indices for each class from ground truth
background_pixels = ~GT;  % Class 0: Clean salt
contaminated_pixels = GT; % Class 1: Sunscreen

% Limit sample size for computational efficiency
max_samples_per_class = min(2000, min(sum(background_pixels(:)), sum(contaminated_pixels(:))));

% Randomly sample from each class
background_indices = find(background_pixels);
contaminated_indices = find(contaminated_pixels);

if length(background_indices) > max_samples_per_class
    background_indices = background_indices(randperm(length(background_indices), max_samples_per_class));
end
if length(contaminated_indices) > max_samples_per_class
    contaminated_indices = contaminated_indices(randperm(length(contaminated_indices), max_samples_per_class));
end

% Extract spectra for sampled pixels
background_spectra = reshaped_cube1(background_indices, valid_bands);
contaminated_spectra = reshaped_cube1(contaminated_indices, valid_bands);

% Preprocess spectra
fprintf('Preprocessing ground truth spectra...\n');
background_processed = preprocess_spectra(background_spectra, wavelengths_clean);
contaminated_processed = preprocess_spectra(contaminated_spectra, wavelengths_clean);

% Create ground truth validation dataset
X_gt = [background_processed; contaminated_processed];
y_gt = [zeros(size(background_processed, 1), 1); 
        ones(size(contaminated_processed, 1), 1)];

% Store pixel indices for spatial evaluation
pixel_indices_gt = [background_indices; contaminated_indices];

fprintf('GROUND TRUTH VALIDATION dataset created:\n');
fprintf('  Total samples: %d\n', size(X_gt, 1));
fprintf('  Class 0 (Background/Salt): %d samples (%.1f%%)\n', ...
    sum(y_gt==0), 100*sum(y_gt==0)/length(y_gt));
fprintf('  Class 1 (Contaminated): %d samples (%.1f%%)\n', ...
    sum(y_gt==1), 100*sum(y_gt==1)/length(y_gt));

%% ========================================================================
%  SECTION 7: VISUALIZE PREPROCESSED SPECTRA
% ========================================================================
% Plot mean spectra after preprocessing to confirm spectral separation
% and identify key fluorescence emission bands
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 7: SPECTRAL VISUALIZATION\n');
fprintf('========================================\n');

%fig2 = figure('Position', [200, 200, 1200, 500], 'Name', 'Preprocessed Spectra');

% Plot preprocessed mean spectra
figure;
%subplot(1,2,1);
plot(wavelengths_clean, mean(salt_processed, 1), 'g-', 'LineWidth', 2.5);
hold on;
plot(wavelengths_clean, mean(sunscreen_processed, 1), 'b-', 'LineWidth', 2.5);
xlabel('Wavelength (nm)', 'FontSize', 11);
ylabel('Normalized Intensity', 'FontSize', 11);
title('Preprocessed Mean Spectra (Training ROIs)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Salt Crystal', 'Sunscreen Residue', 'Location', 'best', 'FontSize', 10);
grid on;

%% Plot difference spectrum
figure;
%subplot(1,2,2);
diff_spectrum = mean(sunscreen_processed, 1) - mean(salt_processed, 1);
plot(wavelengths_clean, diff_spectrum, 'r-', 'LineWidth', 2);
xlabel('Wavelength (nm)', 'FontSize', 11);
ylabel('Difference in Normalized Intensity', 'FontSize', 11);
title('Differential Spectrum (Sunscreen - Salt)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
hold on;
yline(0, 'k--', 'LineWidth', 1);

fprintf('Spectral visualization completed.\n');

%% ========================================================================
%  SECTION 8: HYPERPARAMETER OPTIMIZATION - NUMBER OF COMPONENTS
% ========================================================================
% Determine optimal number of sparse principal components using:
% 1. Scree plot (eigenvalue analysis)
% 2. Cumulative variance explained
% 3. Cross-validated classification accuracy
% USING ROI TRAINING DATA FOR OPTIMIZATION
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 8: COMPONENT NUMBER OPTIMIZATION\n');
fprintf('========================================\n');

% Standardize features for PCA (USING ROI TRAINING DATA)
X_standardized = zscore(X_train_roi);

% Compute regular PCA to analyze variance structure
fprintf('Computing standard PCA for variance analysis...\n');
[coeff_pca, score_pca, latent_pca, ~, explained_pca] = pca(X_standardized);

% Determine number of components to test
max_components = 15;
cumvar = cumsum(explained_pca(1:max_components));

fprintf('\nVariance explained by components:\n');
for k = 1:min(10, max_components)
    fprintf('  Component %2d: %6.2f%% (Cumulative: %6.2f%%)\n', ...
        k, explained_pca(k), cumvar(k));
end

% Create scree plot
fig3 = figure('Position', [300, 300, 1400, 500], 'Name', 'Component Selection');

subplot(1,3,1);
plot(1:max_components, latent_pca(1:max_components), 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
xlabel('Component Number', 'FontSize', 11);
ylabel('Eigenvalue', 'FontSize', 11);
title('Scree Plot', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
hold on;
% Highlight elbow region (typically around 6-8 components for this data)
% xline(4, 'r--', 'LineWidth', 2, 'Label', 'Elbow');

subplot(1,3,2);
plot(1:max_components, cumvar, 'mo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'm');
xlabel('Component Number', 'FontSize', 11);
ylabel('Cumulative Variance Explained (%)', 'FontSize', 11);
title('Cumulative Variance Explained', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
hold on;
yline(90, 'r--', 'LineWidth', 1.5, 'Label', '90% threshold');
yline(95, 'g--', 'LineWidth', 1.5, 'Label', '95% threshold');

% Cross-validation to find optimal number of components (USING ROI DATA)
fprintf('\nPerforming cross-validation for component selection...\n');
k_folds = 5;  % Use 5-fold CV for faster computation
cv_partition = cvpartition(y_train_roi, 'KFold', k_folds);
component_range = 1:12;
cv_accuracy_components = zeros(length(component_range), 1);

for idx = 1:length(component_range)
    num_comp = component_range(idx);
    fold_acc = zeros(k_folds, 1);
    
    for fold = 1:k_folds
        train_idx = training(cv_partition, fold);
        test_idx = test(cv_partition, fold);
        
        % Use first num_comp components from PCA
        X_train_temp = score_pca(train_idx, 1:num_comp);
        X_test_temp = score_pca(test_idx, 1:num_comp);
        
        % Train SVM
        svm_temp = fitcsvm(X_train_temp, y_train_roi(train_idx), ...
            'KernelFunction', 'rbf', 'Standardize', true);
        
        % Evaluate
        y_pred_temp = predict(svm_temp, X_test_temp);
        fold_acc(fold) = sum(y_pred_temp == y_train_roi(test_idx)) / sum(test_idx);
    end
    
    cv_accuracy_components(idx) = mean(fold_acc);
    fprintf('  %2d components: Accuracy = %.2f%%\n', num_comp, cv_accuracy_components(idx)*100);
end

subplot(1,3,3);
plot(component_range, cv_accuracy_components*100, 'go-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'g');
xlabel('Number of Components', 'FontSize', 11);
ylabel('Cross-Validation Accuracy (%)', 'FontSize', 11);
title('Classification Performance vs. Components', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
hold on;
% [max_acc, opt_comp_idx] = max(cv_accuracy_components);
% xline(component_range(opt_comp_idx), 'r--', 'LineWidth', 2, ...
%     'Label', sprintf('Optimal: %d', component_range(opt_comp_idx)));

%% Select optimal number of components
num_components = 3;  % Based on scree plot elbow and >95% variance
fprintf('\n>>> SELECTED NUMBER OF COMPONENTS: %d\n', num_components);
fprintf('>>> Cumulative variance explained: %.2f%%\n', cumvar(num_components));

%% ========================================================================
%  SECTION 9: SPARSITY PARAMETER OPTIMIZATION (CORRECTED)
% ========================================================================
% Optimize the sparsity parameter (lambda) for Sparse PCA
% CORRECTION: Use larger lambda range and normalize loadings before thresholding
% USING ROI TRAINING DATA FOR OPTIMIZATION
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 9: SPARSITY PARAMETER OPTIMIZATION\n');
fprintf('========================================\n');

% Define lambda search grid - MUCH LARGER VALUES
% The lambda must be relative to the loading magnitudes
lambda_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0];
n_lambdas = length(lambda_grid);

% Storage for results
sparsity_levels = zeros(n_lambdas, 1);
lambda_accuracy = zeros(n_lambdas, 1);
avg_nonzero = zeros(n_lambdas, 1);

fprintf('Testing %d lambda values...\n', n_lambdas);

for lambda_idx = 1:n_lambdas
    lambda_test = lambda_grid(lambda_idx);
    
    % Compute Sparse PCA with this lambda (USING ROI TRAINING DATA)
    [spca_coeff_test, spca_score_test] = manual_sparse_pca_corrected(X_standardized, num_components, lambda_test);
    
    % Calculate sparsity (percentage of zero coefficients)
    nonzero_per_comp = sum(spca_coeff_test ~= 0, 1);
    avg_nonzero(lambda_idx) = mean(nonzero_per_comp);
    sparsity_levels(lambda_idx) = 100 * (1 - mean(nonzero_per_comp) / size(spca_coeff_test, 1));
    
    % Evaluate classification performance with 5-fold CV (USING ROI DATA)
    fold_acc_lambda = zeros(k_folds, 1);
    for fold = 1:k_folds
        train_idx = training(cv_partition, fold);
        test_idx = test(cv_partition, fold);
        
        svm_lambda = fitcsvm(spca_score_test(train_idx, :), y_train_roi(train_idx), ...
            'KernelFunction', 'rbf', 'Standardize', true);
        
        y_pred_lambda = predict(svm_lambda, spca_score_test(test_idx, :));
        fold_acc_lambda(fold) = sum(y_pred_lambda == y_train_roi(test_idx)) / sum(test_idx);
    end
    
    lambda_accuracy(lambda_idx) = mean(fold_acc_lambda);
    
    fprintf('  Lambda = %.4f: Sparsity = %5.1f%%, Avg non-zero = %5.1f, Accuracy = %6.2f%%\n', ...
        lambda_test, sparsity_levels(lambda_idx), avg_nonzero(lambda_idx), lambda_accuracy(lambda_idx)*100);
end

% Plot sparsity-performance trade-off
fig4 = figure('Position', [400, 300, 1200, 500], 'Name', 'Sparsity Optimization');

subplot(1,2,1);
yyaxis left
plot(lambda_grid, sparsity_levels, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
ylabel('Sparsity Level (%)', 'FontSize', 11);
ylim([0 100]);
yyaxis right
plot(lambda_grid, avg_nonzero, 'rs-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
ylabel('Avg. Non-zero Coefficients per Component', 'FontSize', 11);
xlabel('Lambda (Sparsity Parameter)', 'FontSize', 11);
title('Sparsity vs. Lambda', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XScale', 'log');
grid on;
legend('Sparsity %', 'Non-zero Coeff.', 'Location', 'best');

subplot(1,2,2);
plot(sparsity_levels, lambda_accuracy*100, 'mo-', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', 'm');
xlabel('Sparsity Level (%)', 'FontSize', 11);
ylabel('Cross-Validation Accuracy (%)', 'FontSize', 11);
title('Sparsity-Performance Trade-off', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
hold on;
% Highlight target sparsity range (60-80%)
patch([60 80 80 60], [min(ylim) min(ylim) max(ylim) max(ylim)], 'g', ...
    'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'Target Range');

% Select optimal lambda - prioritize sparsity while maintaining >99% accuracy
% Find lambda with highest sparsity that maintains accuracy > 99%
valid_lambdas = find(lambda_accuracy >= 0.99);
if ~isempty(valid_lambdas)
    % Among those maintaining 99% accuracy, choose highest sparsity
    [~, best_idx] = max(sparsity_levels(valid_lambdas));
    opt_lambda_idx = valid_lambdas(best_idx);
else
    % If no lambda gives >99%, choose the one with best accuracy
    [~, opt_lambda_idx] = max(lambda_accuracy);
end

optimal_lambda = lambda_grid(opt_lambda_idx);

fprintf('\n>>> SELECTED SPARSITY PARAMETER: Lambda = %.4f\n', optimal_lambda);
fprintf('>>> Achieved sparsity: %.1f%%\n', sparsity_levels(opt_lambda_idx));
fprintf('>>> Average non-zero coefficients: %.1f per component\n', avg_nonzero(opt_lambda_idx));
fprintf('>>> Cross-validation accuracy: %.2f%%\n', lambda_accuracy(opt_lambda_idx)*100);

%% ========================================================================
%  SECTION 10: FINAL SPARSE PCA WITH OPTIMAL PARAMETERS
% ========================================================================
% Apply Sparse PCA with optimized hyperparameters
% Extract sparse principal components for classification
% USING ROI TRAINING DATA FOR FINAL MODEL
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 10: SPARSE PCA FEATURE EXTRACTION\n');
fprintf('========================================\n');

fprintf('Computing Sparse PCA with optimized parameters...\n');
fprintf('  Number of components: %d\n', num_components);
fprintf('  Sparsity parameter (lambda): %.4f\n', optimal_lambda);

% Compute final Sparse PCA (USING ROI TRAINING DATA)
[spca_coeff, spca_score] = manual_sparse_pca_corrected(X_standardized, num_components, optimal_lambda);

% Analyze sparsity of each component
fprintf('\nSparsity analysis per component:\n');
for comp = 1:num_components
    nonzero_count = sum(spca_coeff(:, comp) ~= 0);
    sparsity_pct = 100 * (1 - nonzero_count / size(spca_coeff, 1));
    fprintf('  Component %d: %3d non-zero loadings (%.1f%% sparse)\n', ...
        comp, nonzero_count, sparsity_pct);
end

% Visualize sparse loading patterns
fig5 = figure('Position', [100, 100, 1600, 900], 'Name', 'Sparse PCA Components');
for comp = 1:num_components
    subplot(2, 3, comp);
    stem(wavelengths_clean, spca_coeff(:, comp), 'filled', 'LineWidth', 1.5);
    xlabel('Wavelength (nm)', 'FontSize', 9);
    ylabel('Loading', 'FontSize', 9);
    title(sprintf('Sparse PC %d (%d non-zero)', comp, sum(spca_coeff(:,comp)~=0)), ...
        'FontSize', 10, 'FontWeight', 'bold');
    grid on;
    xlim([wavelengths_clean(1), wavelengths_clean(end)]);
end

fprintf('Sparse PCA completed successfully.\n');

%% Validate Sparse PCA is working correctly
fprintf('\n=== SPARSITY VALIDATION ===\n');
total_coeffs = numel(spca_coeff);
zero_coeffs = sum(spca_coeff(:) == 0);
actual_sparsity = 100 * zero_coeffs / total_coeffs;
fprintf('Total coefficients: %d\n', total_coeffs);
fprintf('Zero coefficients: %d\n', zero_coeffs);
fprintf('Actual sparsity achieved: %.1f%%\n', actual_sparsity);

if actual_sparsity < 10
    warning('Sparsity is very low (%.1f%%). Consider increasing lambda.', actual_sparsity);
end
%% Suppose spca_score is n x k matrix (samples x components)
% Example: plot scores for the first two Sparse PCs

figure;
scatter(spca_score(:,1), spca_score(:,2), 20, 'filled');
xlabel('Sparse PC 1 score');
ylabel('Sparse PC 2 score');
title('Sparse PCA projection: PC1 vs PC2');
grid on;

%% ========================================================================
%  SECTION 11: TRAIN-TEST SPLIT (ROI DATA)
% ========================================================================
% Split ROI dataset into training (80%) and testing (20%) sets
% Use stratified sampling to maintain class proportions
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 11: TRAIN-TEST SPLIT (ROI DATA)\n');
fprintf('========================================\n');

% Set random seed for reproducibility
rng(42);

% Create stratified partition for ROI data
cv_holdout = cvpartition(y_train_roi, 'HoldOut', 0.2);

% Split features and labels (ROI DATA)
X_features_roi = spca_score;  % Use Sparse PCA scores as features
X_train = X_features_roi(training(cv_holdout), :);
X_test_roi = X_features_roi(test(cv_holdout), :);
y_train = y_train_roi(training(cv_holdout));
y_test_roi = y_train_roi(test(cv_holdout));

fprintf('ROI DATASET split completed:\n');
fprintf('  Training set: %d samples (%.1f%%)\n', ...
    length(y_train), 100*length(y_train)/length(y_train_roi));
fprintf('    - Class 0: %d samples\n', sum(y_train==0));
fprintf('    - Class 1: %d samples\n', sum(y_train==1));
fprintf('  Test set: %d samples (%.1f%%)\n', ...
    length(y_test_roi), 100*length(y_test_roi)/length(y_train_roi));
fprintf('    - Class 0: %d samples\n', sum(y_test_roi==0));
fprintf('    - Class 1: %d samples\n', sum(y_test_roi==1));

%% ========================================================================
%  SECTION 12: SVM CLASSIFIER TRAINING
% ========================================================================
% Train Support Vector Machine classifier with RBF kernel
% Use optimized sparse PCA features as input
% TRAINED ON ROI DATA
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 12: SVM CLASSIFIER TRAINING\n');
fprintf('========================================\n');

fprintf('Training SVM classifier on ROI data...\n');
fprintf('  Kernel: RBF (Radial Basis Function)\n');
fprintf('  Box Constraint (C): 1.0\n');
fprintf('  Kernel Scale: Auto-optimized\n');

% Train SVM model on ROI data
svm_model = fitcsvm(X_train, y_train, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'KernelScale', 'auto', ...
    'Standardize', true, ...
    'ClassNames', [0, 1]);

fprintf('SVM training completed.\n');
fprintf('Number of support vectors: %d\n', size(svm_model.SupportVectors, 1));

% Predict on ROI test set
fprintf('\nMaking predictions on ROI test set...\n');
y_pred_roi = predict(svm_model, X_test_roi);

%% ========================================================================
%  SECTION 13: PERFORMANCE EVALUATION - HOLDOUT TEST (ROI DATA)
% ========================================================================
% Evaluate classifier performance on independent ROI test set
% Calculate accuracy, precision, recall, F1-score
% Generate confusion matrix
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 13: HOLDOUT TEST PERFORMANCE (ROI DATA)\n');
fprintf('========================================\n');

% Calculate confusion matrix for ROI test data
conf_mat_roi = confusionmat(y_test_roi, y_pred_roi);

% Extract confusion matrix elements
TN_roi = conf_mat_roi(1,1);  % True Negatives
FP_roi = conf_mat_roi(1,2);  % False Positives
FN_roi = conf_mat_roi(2,1);  % False Negatives
TP_roi = conf_mat_roi(2,2);  % True Positives

% Calculate performance metrics
accuracy_roi = (TP_roi + TN_roi) / (TP_roi + TN_roi + FP_roi + FN_roi);
precision_roi = TP_roi / (TP_roi + FP_roi);
recall_roi = TP_roi / (TP_roi + FN_roi);
f1_score_roi = 2 * (precision_roi * recall_roi) / (precision_roi + recall_roi);

% Display results
fprintf('\n=== ROI HOLDOUT TEST SET RESULTS ===\n');
fprintf('Accuracy:  %.2f%%\n', accuracy_roi * 100);
fprintf('Precision: %.4f\n', precision_roi);
fprintf('Recall:    %.4f\n', recall_roi);
fprintf('F1-Score:  %.4f\n', f1_score_roi);
fprintf('\nConfusion Matrix:\n');
fprintf('                 Predicted\n');
fprintf('                 Salt  Sunscreen\n');
fprintf('Actual Salt      %4d    %4d\n', TN_roi, FP_roi);
fprintf('Actual Sunscreen %4d    %4d\n', FN_roi, TP_roi);
fprintf('\nTrue Positives:   %d\n', TP_roi);
fprintf('False Positives:  %d\n', FP_roi);
fprintf('True Negatives:   %d\n', TN_roi);
fprintf('False Negatives:  %d\n', FN_roi);

%% ========================================================================
%  SECTION 14: K-FOLD CROSS-VALIDATION (ROI DATA)
% ========================================================================
% Perform rigorous k-fold cross-validation to assess generalization
% Report mean performance with standard deviation across folds
% USING ROI TRAINING DATA
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 14: K-FOLD CROSS-VALIDATION (ROI DATA)\n');
fprintf('========================================\n');

% Set up k-fold cross-validation
k_folds_final = 10;
fprintf('Performing %d-fold stratified cross-validation on ROI data...\n', k_folds_final);

cv_partition_final = cvpartition(y_train_roi, 'KFold', k_folds_final);

% Storage for fold results
cv_accuracy = zeros(k_folds_final, 1);
cv_precision = zeros(k_folds_final, 1);
cv_recall = zeros(k_folds_final, 1);
cv_f1 = zeros(k_folds_final, 1);

% Perform cross-validation on ROI data
for fold = 1:k_folds_final
    % Get train/test indices for this fold
    train_idx = training(cv_partition_final, fold);
    test_idx = test(cv_partition_final, fold);
    
    % Train SVM on this fold
    svm_fold = fitcsvm(X_features_roi(train_idx, :), y_train_roi(train_idx), ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', 1, ...
        'KernelScale', 'auto', ...
        'Standardize', true, ...
        'ClassNames', [0, 1]);
    
    % Predict on test fold
    y_pred_fold = predict(svm_fold, X_features_roi(test_idx, :));
    
    % Calculate confusion matrix for this fold
    conf_fold = confusionmat(y_train_roi(test_idx), y_pred_fold);
    TN_fold = conf_fold(1,1);
    FP_fold = conf_fold(1,2);
    FN_fold = conf_fold(2,1);
    TP_fold = conf_fold(2,2);
    
    % Calculate metrics
    cv_accuracy(fold) = (TP_fold + TN_fold) / (TP_fold + TN_fold + FP_fold + FN_fold);
    cv_precision(fold) = TP_fold / (TP_fold + FP_fold);
    cv_recall(fold) = TP_fold / (TP_fold + FN_fold);
    cv_f1(fold) = 2 * (cv_precision(fold) * cv_recall(fold)) / (cv_precision(fold) + cv_recall(fold));
    
    fprintf('  Fold %2d: Accuracy = %.2f%%, Precision = %.4f, Recall = %.4f, F1 = %.4f\n', ...
        fold, cv_accuracy(fold)*100, cv_precision(fold), cv_recall(fold), cv_f1(fold));
end

% Calculate summary statistics
fprintf('\n=== ROI K-FOLD CROSS-VALIDATION SUMMARY ===\n');
fprintf('Accuracy:  %.2f%% ± %.2f%%\n', mean(cv_accuracy)*100, std(cv_accuracy)*100);
fprintf('Precision: %.4f ± %.4f\n', mean(cv_precision), std(cv_precision));
fprintf('Recall:    %.4f ± %.4f\n', mean(cv_recall), std(cv_recall));
fprintf('F1-Score:  %.4f ± %.4f\n', mean(cv_f1), std(cv_f1));

% Visualize cross-validation results
fig6 = figure('Position', [200, 200, 1200, 400], 'Name', 'Cross-Validation Results (ROI Data)');

subplot(1,4,1);
bar(cv_accuracy*100, 'FaceColor', [0.2 0.6 0.8]);
ylabel('Accuracy (%)', 'FontSize', 10);
xlabel('Fold Number', 'FontSize', 10);
title(sprintf('Accuracy per Fold\nMean: %.2f%%', mean(cv_accuracy)*100), 'FontSize', 11);
ylim([min(95, min(cv_accuracy*100)-1), 100]);
grid on;

subplot(1,4,2);
bar(cv_precision, 'FaceColor', [0.8 0.4 0.2]);
ylabel('Precision', 'FontSize', 10);
xlabel('Fold Number', 'FontSize', 10);
title(sprintf('Precision per Fold\nMean: %.4f', mean(cv_precision)), 'FontSize', 11);
ylim([min(0.95, min(cv_precision)-0.01), 1.0]);
grid on;

subplot(1,4,3);
bar(cv_recall, 'FaceColor', [0.4 0.8 0.4]);
ylabel('Recall', 'FontSize', 10);
xlabel('Fold Number', 'FontSize', 10);
title(sprintf('Recall per Fold\nMean: %.4f', mean(cv_recall)), 'FontSize', 11);
ylim([min(0.95, min(cv_recall)-0.01), 1.0]);
grid on;

subplot(1,4,4);
bar(cv_f1, 'FaceColor', [0.8 0.6 0.8]);
ylabel('F1-Score', 'FontSize', 10);
xlabel('Fold Number', 'FontSize', 10);
title(sprintf('F1-Score per Fold\nMean: %.4f', mean(cv_f1)), 'FontSize', 11);
ylim([min(0.95, min(cv_f1)-0.01), 1.0]);
grid on;

%% ========================================================================
%  SECTION 15: VISUALIZE CLASSIFICATION IN FEATURE SPACE (ROI DATA)
% ========================================================================
% Visualize data separation in the first two sparse principal components
% Display decision boundary and support vectors
% USING ROI DATA
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 15: FEATURE SPACE VISUALIZATION (ROI DATA)\n');
fprintf('========================================\n');

fig7 = figure('Position', [300, 300, 1400, 500], 'Name', 'Classification Visualization (ROI Data)');

% Plot 1: Class separation in first 2 components
subplot(1,3,1);
gscatter(X_features_roi(:,1), X_features_roi(:,2), y_train_roi, 'gb', 'o*', 8);
xlabel('Sparse PC 1', 'FontSize', 11);
ylabel('Sparse PC 2', 'FontSize', 11);
title('Class Separation (First 2 Components)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Salt Crystal', 'Sunscreen Residue', 'Location', 'best', 'FontSize', 10);
grid on;
axis tight;

% Plot 2: Feature importance (top 10 wavelengths from Component 1)
subplot(1,3,2);
[sorted_loadings, sorted_idx] = sort(abs(spca_coeff(:,1)), 'descend');
top_n = 10;
top_wavelengths = wavelengths_clean(sorted_idx(1:top_n));
top_loadings = sorted_loadings(1:top_n);

barh(top_wavelengths, top_loadings, 'FaceColor', [0.3 0.6 0.9]);
xlabel('Absolute Loading Value', 'FontSize', 11);
ylabel('Wavelength (nm)', 'FontSize', 11);
title(sprintf('Top %d Most Important Wavelengths\n(Component 1)', top_n), ...
    'FontSize', 12, 'FontWeight', 'bold');
grid on;
set(gca, 'YDir', 'reverse');

% Plot 3: Confusion matrix heatmap - CORRECTED
subplot(1,3,3);
cm = confusionchart(y_test_roi, y_pred_roi);
cm.FontSize = 10;
% Don't use title() with confusionchart - set property directly
cm.Title = 'Confusion Matrix (ROI Test Set)';
cm.XLabel = 'Predicted Class';
cm.YLabel = 'True Class';

fprintf('Feature space visualization completed.\n');

%% ========================================================================
%  SECTION 16: FULL-IMAGE PIXEL-WISE CLASSIFICATION
% ========================================================================
% Apply trained SVM classifier to every pixel in the hyperspectral image
% Generate spatial map of contamination distribution
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 16: FULL-IMAGE CLASSIFICATION\n');
fprintf('========================================\n');

fprintf('Applying classifier to entire hyperspectral image...\n');
fprintf('Total pixels to classify: %d\n', size(Cuberead1,1) * size(Cuberead1,2));

% Initialize classification map
classification_map = zeros(size(Cuberead1, 1), size(Cuberead1, 2));

% Process in batches for memory efficiency
batch_size = 1000;
total_pixels = size(Cuberead1, 1) * size(Cuberead1, 2);

% Reshape entire cube
full_reshaped = reshape(Cuberead1, total_pixels, num_bands);
full_reshaped_clean = full_reshaped(:, valid_bands);

fprintf('Processing pixels in batches of %d...\n', batch_size);

% Calculate standardization parameters from ROI training data
X_mean = mean(X_train_roi, 1);
X_std = std(X_train_roi, 0, 1);

for pixel_start = 1:batch_size:total_pixels
    pixel_end = min(pixel_start + batch_size - 1, total_pixels);
    
    % Extract batch
    batch_spectra = full_reshaped_clean(pixel_start:pixel_end, :);
    
    % Preprocess batch
    batch_processed = preprocess_spectra(batch_spectra, wavelengths_clean);
    
    % Standardize using ROI training statistics and transform using Sparse PCA
    batch_standardized = (batch_processed - X_mean) ./ X_std;
    batch_features = batch_standardized * spca_coeff;
    
    % Classify batch
    batch_pred = predict(svm_model, batch_features);
    
    % Store results in 2D map
    [rows, cols] = ind2sub([size(Cuberead1,1), size(Cuberead1,2)], pixel_start:pixel_end);
    for i = 1:length(rows)
        classification_map(rows(i), cols(i)) = batch_pred(i);
    end
    
    % Progress update
    if mod(pixel_start, 50000) == 1 || pixel_end == total_pixels
        fprintf('  Processed %d/%d pixels (%.1f%%)\n', ...
            pixel_end, total_pixels, (pixel_end/total_pixels)*100);
    end
end

% Calculate contamination statistics
sunscreen_pixels_total = sum(classification_map(:) == 1);
sunscreen_percentage = (sunscreen_pixels_total / total_pixels) * 100;

fprintf('\n=== FULL-IMAGE CLASSIFICATION COMPLETE ===\n');
fprintf('Total pixels classified: %d\n', total_pixels);
fprintf('Sunscreen-contaminated pixels: %d (%.2f%%)\n', ...
    sunscreen_pixels_total, sunscreen_percentage);
fprintf('Clean salt pixels: %d (%.2f%%)\n', ...
    total_pixels - sunscreen_pixels_total, 100 - sunscreen_percentage);

%% ========================================================================
%  SECTION 17: GROUND TRUTH VALIDATION
% ========================================================================
% Evaluate classifier performance against comprehensive ground truth
% Calculate spatial accuracy, precision, recall, F1-score
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 17: GROUND TRUTH VALIDATION\n');
fprintf('========================================\n');

% Create ground truth classification map
GT_class_map = zeros(size(classification_map));
GT_class_map(GT) = 1;

% Calculate confusion matrix for entire image
TP_spatial = sum(classification_map(:) == 1 & GT_class_map(:) == 1);
FP_spatial = sum(classification_map(:) == 1 & GT_class_map(:) == 0);
TN_spatial = sum(classification_map(:) == 0 & GT_class_map(:) == 0);
FN_spatial = sum(classification_map(:) == 0 & GT_class_map(:) == 1);

% Calculate performance metrics
accuracy_spatial = (TP_spatial + TN_spatial) / (TP_spatial + TN_spatial + FP_spatial + FN_spatial);
precision_spatial = TP_spatial / (TP_spatial + FP_spatial);
recall_spatial = TP_spatial / (TP_spatial + FN_spatial);
f1_score_spatial = 2 * (precision_spatial * recall_spatial) / (precision_spatial + recall_spatial);

% Display spatial performance
fprintf('\n=== SPATIAL PERFORMANCE (vs. GROUND TRUTH) ===\n');
fprintf('True Positives:  %d\n', TP_spatial);
fprintf('False Positives: %d\n', FP_spatial);
fprintf('True Negatives:  %d\n', TN_spatial);
fprintf('False Negatives: %d\n', FN_spatial);
fprintf('\nPerformance Metrics:\n');
fprintf('  Accuracy:  %.4f (%.2f%%)\n', accuracy_spatial, accuracy_spatial*100);
fprintf('  Precision: %.4f\n', precision_spatial);
fprintf('  Recall:    %.4f\n', recall_spatial);
fprintf('  F1-Score:  %.4f\n', f1_score_spatial);

% Also evaluate on the sampled ground truth dataset
fprintf('\n=== SAMPLED GROUND TRUTH VALIDATION ===\n');

% Transform ground truth data using the same Sparse PCA
X_gt_standardized = (X_gt - X_mean) ./ X_std;
X_gt_features = X_gt_standardized * spca_coeff;

% Predict on ground truth samples
y_pred_gt = predict(svm_model, X_gt_features);

% Calculate performance on ground truth samples
conf_mat_gt = confusionmat(y_gt, y_pred_gt);
TN_gt = conf_mat_gt(1,1);
FP_gt = conf_mat_gt(1,2);
FN_gt = conf_mat_gt(2,1);
TP_gt = conf_mat_gt(2,2);

accuracy_gt = (TP_gt + TN_gt) / (TP_gt + TN_gt + FP_gt + FN_gt);
precision_gt = TP_gt / (TP_gt + FP_gt);
recall_gt = TP_gt / (TP_gt + FN_gt);
f1_score_gt = 2 * (precision_gt * recall_gt) / (precision_gt + recall_gt);

fprintf('Accuracy:  %.2f%%\n', accuracy_gt * 100);
fprintf('Precision: %.4f\n', precision_gt);
fprintf('Recall:    %.4f\n', recall_gt);
fprintf('F1-Score:  %.4f\n', f1_score_gt);

%% ========================================================================
%  SECTION 18: VISUALIZE FINAL CLASSIFICATION RESULTS WITH GROUND TRUTH
% ========================================================================
% Display original image with ROIs alongside pixel-wise classification map
% and ground truth comparison
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 18: FINAL VISUALIZATION WITH GROUND TRUTH\n');
fprintf('========================================\n');

fig8 = figure('Position', [100, 100, 1800, 600], 'Name', 'Final Classification Results with Ground Truth');

% % Original image with ROIs
% subplot(1,3,1);
% imagesc(monoImage);
% colormap(gca, gray);
% hold on;
% plot(saltCrystal_boundary{1}(:,2), saltCrystal_boundary{1}(:,1), 'g', 'LineWidth', 2.5);
% plot(Sunscreenresidues_boundary{1}(:,2), Sunscreenresidues_boundary{1}(:,1), 'b', 'LineWidth', 2.5);
% title('Original Image with Training ROIs', 'FontSize', 12, 'FontWeight', 'bold');
% legend('Salt Crystal ROI', 'Sunscreen ROI', 'Location', 'best', 'FontSize', 10);
% xlabel('X (pixels)', 'FontSize', 10);
% ylabel('Y (pixels)', 'FontSize', 10);
% axis image;
% colorbar;

% Classification map
%subplot(1,3,2);
figure;
imagesc(classification_map);
colormap(gca, [0 1 0; 0 0 1]);  % Green for salt, Blue for sunscreen
cb = colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'Clean Salt', 'Sunscreen Contaminated'});
cb.FontSize = 10;
title(sprintf('SVM Classification Map\n%.1f%% Contaminated', sunscreen_percentage), ...
    'FontSize', 12, 'FontWeight', 'bold');
xlabel('X (pixels)', 'FontSize', 10);
ylabel('Y (pixels)', 'FontSize', 10);
axis image;

% Ground truth comparison
%subplot(1,3,3);
% Create difference map: 0=TN, 1=TP, 2=FP, 3=FN
figure;
difference_map = zeros(size(classification_map));
difference_map(classification_map == 0 & GT_class_map == 0) = 0; % TN
difference_map(classification_map == 1 & GT_class_map == 1) = 1; % TP
difference_map(classification_map == 1 & GT_class_map == 0) = 2; % FP
difference_map(classification_map == 0 & GT_class_map == 1) = 3; % FN

imagesc(difference_map);
colormap(gca, [0 1 0; 0 0 1; 1 1 0; 1 0 0]); % Green, Blue, Yellow, Red
colorbar('Ticks', [0.375, 1.125, 1.875, 2.625], ...
         'TickLabels', {'TN (Correct Salt)', 'TP (Correct Contam.)', ...
                       'FP (False Alarm)', 'FN (Missed Contam.)'});
title(sprintf('Classification vs. Ground Truth\nAccuracy: %.1f%%', accuracy_spatial*100), ...
    'FontSize', 12, 'FontWeight', 'bold');
xlabel('X (pixels)', 'FontSize', 10);
ylabel('Y (pixels)', 'FontSize', 10);
axis image;

fprintf('Final visualization completed.\n');
%%
%% ========================================================================
%  SECTION 13: PERFORMANCE EVALUATION VS AUTOMATED GROUND TRUTH
% ========================================================================
fprintf('\n========================================\n');
fprintf('SECTION 13: PERFORMANCE EVALUATION VS AUTOMATED GROUND TRUTH\n');
fprintf('========================================\n');

% --- Load prediction and ground truth ---
% prediction_map: the SVM result (should be same shape as GT)
% GT: binary mask from ground truth (1=contaminated, 0=clean salt)

prediction_map = classification_map;  % Use your classifier output

% --- Convert labels to the same type for confusionmat ---
gt_labels = double(GT(:));           % Ground truth (0/1), ensure numeric type
pred_labels = double(prediction_map(:));  % Prediction (0/1), ensure numeric type

% --- Confusion matrix ---
cm = confusionmat(gt_labels, pred_labels);

% Extract confusion elements (binary)
TP = cm(2,2);   % True Positive: contaminated region predicted as contaminated
TN = cm(1,1);   % True Negative: clean region predicted as clean
FP = cm(1,2);   % False Positive: clean predicted contaminated
FN = cm(2,1);   % Contaminated predicted clean

% --- Metrics ---
accuracy = (TP + TN) / (TP + TN + FP + FN);
recall = TP / (TP + FN);         % Sensitivity
precision = TP / (TP + FP);
f1score = 2 * (precision * recall) / (precision + recall);

% --- Display results ---
fprintf('Confusion Matrix:\n');
disp(cm);
fprintf('Accuracy   : %.3f\n', accuracy);
fprintf('Recall     : %.3f\n', recall);
fprintf('Precision  : %.3f\n', precision);
fprintf('F1 Score   : %.3f\n', f1score);
fprintf('True Positives: %d\n', TP);
fprintf('True Negatives: %d\n', TN);
fprintf('False Positives: %d\n', FP);
fprintf('False Negatives: %d\n', FN);

% --- Optional: Visualize comparison ---
figure;
subplot(1,2,1);
imagesc(GT);
title('Automated Ground Truth (1 = contaminated)');
colormap([1 1 1; 0 0 1]); axis image; colorbar;

subplot(1,2,2);
imagesc(prediction_map);
title('SVM Classification Result (1 = contaminated)');
colormap([1 1 1; 0 0 1]); axis image; colorbar;
%% ========================================================================

%% ========================================================================
%  SECTION 13: PERFORMANCE EVALUATION VS AUTOMATED GROUND TRUTH (FULL IMAGE)
% ========================================================================
fprintf('\n========================================\n');
fprintf('SECTION 13: PERFORMANCE EVALUATION VS AUTOMATED GROUND TRUTH\n');
fprintf('========================================\n');

% prediction_map: classifier output for all image pixels (0/1)
% GT: automated ground truth mask for all pixels (1 = contaminated, 0 = clean salt)

gt_labels = double(GT(:));                     % Ground truth for all pixels
pred_labels = double(prediction_map(:));       % Prediction for all pixels

cm = confusionmat(gt_labels, pred_labels);

TP = cm(2,2);   
TN = cm(1,1);   
FP = cm(1,2);   
FN = cm(2,1);

accuracy  = (TP + TN) / (TP + TN + FP + FN);
recall    = TP / (TP + FN);         % Sensitivity
precision = TP / (TP + FP);
f1score   = 2 * (precision * recall) / (precision + recall);

% Report results (nicely aligned)
fprintf('Confusion Matrix:\n');
fprintf('      %8d %8d\n', cm(1,1), cm(1,2));
fprintf('      %8d %8d\n', cm(2,1), cm(2,2));
fprintf('\n');

fprintf('Accuracy   : %.3f\n', accuracy);
fprintf('Recall     : %.3f\n', recall);
fprintf('Precision  : %.3f\n', precision);
fprintf('F1 Score   : %.3f\n', f1score);
fprintf('True Positives : %d\n', TP);
fprintf('True Negatives : %d\n', TN);
fprintf('False Positives: %d\n', FP);
fprintf('False Negatives: %d\n', FN);

% Optionally, display summary visually as a text box in figure
figure;
imagesc(GT); colormap([1 1 1; 0 0 1]); axis image;
title('Ground Truth (1 = contaminated)');
text(10,10, sprintf('Accuracy: %.3f\nRecall: %.3f\nPrecision: %.3f\nF1: %.3f', ...
    accuracy, recall, precision, f1score), ...
    'FontSize',12,'Color','black','BackgroundColor','white');
colorbar('Ticks',[0,1],'TickLabels',{'Clean Salt','Sunscreen Contaminated'});


%% ========================================================================
%  SECTION 13B: FULL IMAGE PERFORMANCE METRICS VISUALIZATION
% ========================================================================
fprintf('\n========================================\n');
fprintf('SECTION 13B: FULL IMAGE PERFORMANCE METRICS VISUALIZATION\n');
fprintf('========================================\n');

% Assume prediction_map is your classifier output for the entire image (0/1)
% GT is your automated ground truth mask (same size, logical or 0/1)

gt_labels = double(GT(:));
pred_labels = double(prediction_map(:));

cm = confusionmat(gt_labels, pred_labels);
TP = cm(2,2); TN = cm(1,1); FP = cm(1,2); FN = cm(2,1);

accuracy  = (TP + TN) / (TP + TN + FP + FN);
recall    = TP / (TP + FN);         
precision = TP / (TP + FP);
f1score   = 2 * (precision * recall) / (precision + recall);

% Create arrays for unified plotting (1 fold, same style as cross-validation)
accuracy_full  = accuracy * 100;
recall_full    = recall;
precision_full = precision;
f1_full        = f1score;

figure;
subplot(2,2,1);
bar(1, accuracy_full, 'FaceColor', [0 0.5 1]);
ylim([95 100]);
title(sprintf('Accuracy\n%.2f%%', accuracy_full));
xlabel('Fold Number'); ylabel('Accuracy (%)'); set(gca,'XTick',1,'XTickLabel','Full Image'); grid on;

subplot(2,2,2);
bar(1, precision_full, 'FaceColor', [0.85 0.45 0.1]);
ylim([0.95 1]);
title(sprintf('Precision\n%.4f', precision_full));
xlabel('Fold Number'); ylabel('Precision'); set(gca,'XTick',1,'XTickLabel','Full Image'); grid on;

subplot(2,2,3);
bar(1, recall_full, 'FaceColor', [0.1 0.85 0.4]);
ylim([0.95 1]);
title(sprintf('Recall\n%.4f', recall_full));
xlabel('Fold Number'); ylabel('Recall'); set(gca,'XTick',1,'XTickLabel','Full Image'); grid on;

subplot(2,2,4);
bar(1, f1_full, 'FaceColor', [0.7 0.4 0.7]);
ylim([0.95 1]);
title(sprintf('F1-Score\n%.4f', f1_full));
xlabel('Fold Number'); ylabel('F1-Score'); set(gca,'XTick',1,'XTickLabel','Full Image'); grid on;

%% ========================================================================
%  SECTION 13C: CONFUSION MATRIX VISUALIZATION FOR FULL-IMAGE GROUND TRUTH
% ========================================================================

% After calculating your confusion matrix (cm), use MATLAB's confusionchart:
gt_labels = double(GT(:));           % Automated ground truth for all pixels
pred_labels = double(prediction_map(:));   % SVM classifications

% Create categorical labels for better chart visualization
trueCats = categorical(gt_labels, [0 1], {'Clean Salt', 'Sunscreen Contaminated'});
predCats = categorical(pred_labels, [0 1], {'Clean Salt', 'Sunscreen Contaminated'});

cm_chart = confusionchart(trueCats, predCats);
cm_chart.Title = 'Confusion Matrix (Full Image Ground Truth)';
cm_chart.ColumnSummary = 'column-normalized';
cm_chart.RowSummary = 'row-normalized';
cm_chart.FontSize = 12;

% Now show the key metrics as text overlay (above or beside the confusion matrix)
textStr = sprintf('Accuracy: %.3f\nRecall: %.3f\nPrecision: %.3f\nF1: %.3f', ...
    accuracy, recall, precision, f1score);

% You can add annotation box in MATLAB for clarity
annotation('textbox', [0.72, 0.10, 0.15, 0.25], 'String', textStr, ...
    'FontSize', 12, 'BackgroundColor', 'w', ...
    'FitBoxToText', 'on', 'EdgeColor', 'k');

% (Adjust the annotation position [x y w h] as needed for your screen)

%% ========================================================================
%  SECTION: SAVE CLASSIFICATION MAP AS GROUND TRUTH
% ========================================================================
fprintf('\n========================================\n');
fprintf('SECTION: SAVE CLASSIFICATION MAP AS GROUND TRUTH\n');
fprintf('========================================\n');

% Assume classification_map already exists and matches your image size
% (For example, shape [rows, cols], values: 1=clean salt, 2=contaminated)

% 1. Save the full classification map as ground truth
GT_map = classification_map;  % Save as ground truth (multi-class)
save('GT_ClassificationMap1.mat', 'GT_map');
disp('Classification map has been saved as ground truth in GT_ClassificationMap.mat');

% 2. (Optional) Create and save a binary mask: contaminated region only
GT_binary = (GT_map == 1);   % Logical mask: 1=contaminated, 0=clean
save('GT_mask_binary.mat', 'GT_binary');
disp('Binary contaminated region mask saved as GT_mask_binary.mat');

% 3. Visualize the multi-class ground truth
figure;
imagesc(GT_map);
colormap([0 1 0; 0 0 1]);  % Green for class 1 (clean salt), Blue for class 2 (contaminated)
colorbar('Ticks', [1, 2], 'TickLabels', {'Clean Salt', 'Sunscreen Contaminated'});
title('Ground Truth (Classification Map)');
axis image;

% 4. Visualize the binary mask (contaminated region only)
figure;
imagesc(GT_binary);
colormap([1 1 1; 0 0 1]);  % White for clean, Blue for contaminated
colorbar('Ticks', [0, 1], 'TickLabels', {'Clean Salt', 'Sunscreen Contaminated'});
title('Ground Truth Mask (1 = contaminated)');
axis image;



%%
% Ground truth comparison
figure;
% Create difference map: 0=TN, 1=TP, 2=FP, 3=FN
difference_map = zeros(size(classification_map));
difference_map(classification_map == 0 & GT_class_map == 0) = 0; % TN
difference_map(classification_map == 1 & GT_class_map == 1) = 1; % TP
difference_map(classification_map == 1 & GT_class_map == 0) = 2; % FP
difference_map(classification_map == 0 & GT_class_map == 1) = 3; % FN

imagesc(difference_map);
colormap(gca, [0 1 0; 0 0 1; 1 1 0; 1 0 0]); % Green, Blue, Yellow, Red
colorbar('Ticks', [0.375, 1.125, 1.875, 2.625], ...
         'TickLabels', {'TN (Correct Salt)', 'TP (Correct Contam.)', ...
                       'FP (False Alarm)', 'FN (Missed Contam.)'});
title(sprintf('Classification vs. Ground Truth\nAccuracy: %.1f%%', accuracy_spatial*100), ...
    'FontSize', 12, 'FontWeight', 'bold');
xlabel('X (pixels)', 'FontSize', 10);
ylabel('Y (pixels)', 'FontSize', 10);
axis image;

fprintf('Final visualization completed.\n');
%% ========================================================================
%  SECTION 19: SAVE TRAINED MODEL AND RESULTS
% ========================================================================
% Save all necessary components for future use on new samples
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 19: SAVE MODEL AND RESULTS\n');
fprintf('========================================\n');

model_filename = 'trained_sunscreen_model_with_GT.mat';
fprintf('Saving trained model to: %s\n', model_filename);

save(model_filename, ...
    'svm_model', 'spca_coeff', 'wavelengths_clean', 'valid_bands', ...
    'num_components', 'optimal_lambda', ...
    'X_train_roi', 'y_train_roi', 'X_gt', 'y_gt', ...
    'classification_map', 'GT_class_map', ...
    'accuracy_roi', 'precision_roi', 'recall_roi', 'f1_score_roi', ...
    'accuracy_spatial', 'precision_spatial', 'recall_spatial', 'f1_score_spatial', ...
    'accuracy_gt', 'precision_gt', 'recall_gt', 'f1_score_gt', ...
    'cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1', ...
    'X_mean', 'X_std', ...  % Save standardization parameters
    '-v7.3');

fprintf('Model saved successfully.\n');
fprintf('Use this model to detect sunscreen in new hyperspectral cubes.\n');
%% ========================================================================
%  SECTION 19: APPLY MODEL TO NEW SAMPLE (OPTIONAL)
% ========================================================================
% Demonstrate how to use the trained model on a new independent sample
% ========================================================================

fprintf('\n========================================\n');
fprintf('SECTION 19: NEW SAMPLE DETECTION (OPTIONAL)\n');
fprintf('========================================\n');

% Check if new cube file exists
new_cube_file = 'SL2.cube';
if exist(new_cube_file, 'file')
    fprintf('New cube file found: %s\n', new_cube_file);
    fprintf('Applying trained model...\n');
    detect_sunscreen_in_new_cube(new_cube_file);
else
    fprintf('No new cube file specified. Skipping independent validation.\n');
    fprintf('To test on new sample, ensure file exists: %s\n', new_cube_file);
end
%% ========================================================================
%  SECTION 20: GENERATE COMPREHENSIVE SUMMARY REPORT
% ========================================================================
% Print comprehensive summary of all results including ground truth validation
% ========================================================================

fprintf('\n\n');
fprintf('========================================================================\n');
fprintf('                    PROCESSING COMPLETE - SUMMARY REPORT\n');
fprintf('========================================================================\n');
fprintf('\n--- DATA INFORMATION ---\n');
fprintf('Hyperspectral cube: %d x %d x %d\n', size(Cuberead1,1), size(Cuberead1,2), size(Cuberead1,3));
fprintf('Wavelength range: %.0f - %.0f nm\n', wavelengths(1), wavelengths(end));
fprintf('Spectral resolution: %.2f nm\n', spectral_resolution);
fprintf('\n--- PREPROCESSING ---\n');
fprintf('Laser bands removed: %d-%d nm (%d bands)\n', ...
    wavelengths(laser_band_indices(1)), wavelengths(laser_band_indices(end)), length(laser_band_indices));
fprintf('Retained spectral bands: %d\n', length(valid_bands));
fprintf('\n--- TRAINING DATASET (ROI-BASED) ---\n');
fprintf('Total labeled pixels: %d\n', length(y_train_roi));
fprintf('Class 0 (Salt): %d pixels\n', sum(y_train_roi==0));
fprintf('Class 1 (Sunscreen): %d pixels\n', sum(y_train_roi==1));
fprintf('\n--- GROUND TRUTH VALIDATION DATASET ---\n');
fprintf('Total validation pixels: %d\n', length(y_gt));
fprintf('Class 0 (Background): %d pixels\n', sum(y_gt==0));
fprintf('Class 1 (Contaminated): %d pixels\n', sum(y_gt==1));
fprintf('\n--- SPARSE PCA PARAMETERS ---\n');
fprintf('Number of components: %d (explaining %.1f%% variance)\n', num_components, cumvar(num_components));
fprintf('Sparsity parameter (lambda): %.4f\n', optimal_lambda);
fprintf('Average sparsity: %.1f%%\n', sparsity_levels(opt_lambda_idx));
fprintf('\n--- CLASSIFICATION PERFORMANCE (ROI DATA) ---\n');
fprintf('Holdout Test (20%%):\n');
fprintf('  Accuracy:  %.2f%%\n', accuracy_roi*100);
fprintf('  Precision: %.4f\n', precision_roi);
fprintf('  Recall:    %.4f\n', recall_roi);
fprintf('  F1-Score:  %.4f\n', f1_score_roi);
fprintf('\n%d-Fold Cross-Validation (ROI):\n', k_folds_final);
fprintf('  Accuracy:  %.2f%% ± %.2f%%\n', mean(cv_accuracy)*100, std(cv_accuracy)*100);
fprintf('  Precision: %.4f ± %.4f\n', mean(cv_precision), std(cv_precision));
fprintf('  Recall:    %.4f ± %.4f\n', mean(cv_recall), std(cv_recall));
fprintf('  F1-Score:  %.4f ± %.4f\n', mean(cv_f1), std(cv_f1));
fprintf('\n--- GROUND TRUTH VALIDATION PERFORMANCE ---\n');
fprintf('Spatial Accuracy (Full Image):  %.2f%%\n', accuracy_spatial*100);
fprintf('Spatial Precision: %.4f\n', precision_spatial);
fprintf('Spatial Recall:    %.4f\n', recall_spatial);
fprintf('Spatial F1-Score:  %.4f\n', f1_score_spatial);
fprintf('\nSampled Ground Truth Validation:\n');
fprintf('  Accuracy:  %.2f%%\n', accuracy_gt*100);
fprintf('  Precision: %.4f\n', precision_gt);
fprintf('  Recall:    %.4f\n', recall_gt);
fprintf('  F1-Score:  %.4f\n', f1_score_gt);
fprintf('\n--- FULL-IMAGE CLASSIFICATION ---\n');
fprintf('Sunscreen contamination: %.2f%% of image area\n', sunscreen_percentage);
fprintf('Clean salt: %.2f%% of image area\n', 100 - sunscreen_percentage);
fprintf('\n--- ERROR ANALYSIS ---\n');
fprintf('True Positives:  %d pixels\n', TP_spatial);
fprintf('False Positives: %d pixels\n', FP_spatial);
fprintf('False Negatives: %d pixels\n', FN_spatial);
fprintf('True Negatives:  %d pixels\n', TN_spatial);
fprintf('\n========================================================================\n');
fprintf('All results saved to: %s\n', model_filename);
fprintf('========================================================================\n\n');

%% ========================================================================
%  HELPER FUNCTION 1: SPECTRAL PREPROCESSING
% ========================================================================
% Applies complete preprocessing pipeline to fluorescence spectra
% Steps: baseline correction, clipping, normalization, smoothing
% ========================================================================

function processed_spectra = preprocess_spectra(spectra, wavelengths)
    % Remove laser region (already handled in main code, but double-check)
    laser_mask = wavelengths >= 440 & wavelengths <= 460;
    if any(laser_mask)
        spectra_clean = spectra(:, ~laser_mask);
        wavelengths_clean = wavelengths(~laser_mask);
    else
        spectra_clean = spectra;
        wavelengths_clean = wavelengths;
    end
    
    % Step 1: Baseline correction (subtract 10th percentile)
    baseline = prctile(spectra_clean, 10, 2);
    spectra_corrected = spectra_clean - baseline;
    
    % Step 2: Clip negative values (fluorescence is non-negative)
    spectra_corrected(spectra_corrected < 0) = 0;
    
    % Step 3: Area normalization (normalize by total integrated intensity)
    area = trapz(wavelengths_clean, spectra_corrected, 2);
    area(area == 0) = 1;  % Avoid division by zero
    spectra_normalized = spectra_corrected ./ area;
    
    % Step 4: Savitzky-Golay smoothing (polynomial order 3, window 11)
    for i = 1:size(spectra_normalized, 1)
        if length(spectra_normalized(i,:)) >= 11
            spectra_normalized(i,:) = sgolayfilt(spectra_normalized(i,:), 3, 11);
        end
    end
    
    processed_spectra = spectra_normalized;
end

%% ========================================================================
%  CORRECTED HELPER FUNCTION: SPARSE PCA WITH PROPER THRESHOLDING
% ========================================================================

function [sparse_coeff, sparse_scores] = manual_sparse_pca_corrected(X, k, lambda)
    % CORRECTED Sparse PCA implementation with proper soft thresholding
    % Key fix: Normalize loadings BEFORE applying soft threshold
    
    [n, p] = size(X);
    sparse_coeff = zeros(p, k);
    sparse_scores = zeros(n, k);
    
    X_current = X;
    
    for comp = 1:k
        % Initialize with standard PCA
        [~, ~, v] = svd(X_current, 'econ');
        v_init = v(:, 1);
        
        % Power iteration with soft thresholding
        max_iter = 200;  % Increased iterations
        tol = 1e-6;
        v = v_init;
        
        for iter = 1:max_iter
            v_old = v;
            
            % Power iteration step
            z = X_current' * (X_current * v);
            
            % CRITICAL FIX: Scale lambda relative to max magnitude
            threshold = lambda * max(abs(z));
            
            % Soft thresholding with scaled threshold
            v = sign(z) .* max(abs(z) - threshold, 0);
            
            % Normalize (if not all zeros)
            v_norm = norm(v);
            if v_norm > 1e-10
                v = v / v_norm;
            else
                % If all zeros, use previous iteration or reinitialize
                fprintf('  Warning: All-zero loading in component %d, iteration %d\n', comp, iter);
                v = v_old;
                break;
            end
            
            % Check convergence
            if norm(v - v_old) < tol
                break;
            end
        end
        
        % Store sparse loading vector
        sparse_coeff(:, comp) = v;
        
        % Calculate scores (projection)
        scores = X_current * v;
        sparse_scores(:, comp) = scores;
        
        % Deflate data matrix
        X_current = X_current - scores * v';
        
        % Report sparsity
        nonzero_count = sum(v ~= 0);
        fprintf('  Component %d: %d non-zero loadings (%.1f%% sparse)\n', ...
            comp, nonzero_count, 100*(1-nonzero_count/p));
    end
end
%% ========================================================================
%  HELPER FUNCTION 3: APPLY MODEL TO NEW CUBE
% ========================================================================
% Loads a new hyperspectral cube and applies the trained classifier
% Generates classification map for the new sample
% ========================================================================

function detect_sunscreen_in_new_cube(new_cube_filename)
    fprintf('\n--- APPLYING MODEL TO NEW SAMPLE ---\n');
    fprintf('Loading: %s\n', new_cube_filename);
    
    % Load trained model
    if ~exist('trained_sunscreen_model_with_GT.mat', 'file')
        error('Trained model not found. Run training pipeline first.');
    end
    
    load('trained_sunscreen_model.mat', 'svm_model', 'spca_coeff', ...
        'wavelengths_clean', 'valid_bands', 'X');
    
    % Load new cube
    Cuberead_new = multibandread(new_cube_filename, [696, 520, 128], 'int16', 32768, 'bil', 'ieee-le');
    Cuberead_new = permute(Cuberead_new, [2, 1, 3]);
    fprintf('New cube loaded: %d x %d x %d\n', size(Cuberead_new,1), size(Cuberead_new,2), size(Cuberead_new,3));
    
    % Process entire cube
    total_pixels = size(Cuberead_new,1) * size(Cuberead_new,2);
    reshaped_new = reshape(Cuberead_new, total_pixels, size(Cuberead_new,3));
    new_spectra_clean = reshaped_new(:, valid_bands);
    
    % Preprocess
    fprintf('Preprocessing new sample spectra...\n');
    new_processed = preprocess_spectra(new_spectra_clean, wavelengths_clean);
    
    % Standardize and transform
    new_standardized = (new_processed - mean(X, 1)) ./ std(X, 0, 1);
    new_features = new_standardized * spca_coeff;
    
    % Predict
    fprintf('Classifying pixels...\n');
    new_predictions = predict(svm_model, new_features);
    
    % Create detection map
    detection_map = reshape(new_predictions, size(Cuberead_new,1), size(Cuberead_new,2));
    
    % Calculate statistics
    sunscreen_pct = (sum(new_predictions == 1) / length(new_predictions)) * 100;
    
    % Visualize results
    figure('Position', [100, 100, 1400, 600], 'Name', 'New Sample Detection');
    
    subplot(1,2,1);
    imagesc(Cuberead_new(:,:,24));
    title('New Sample - Original Image (Band 24)', 'FontSize', 12, 'FontWeight', 'bold');
    colormap(gca, jet);
    colorbar;
    axis image;
    
    subplot(1,2,2);
    imagesc(detection_map);
    colormap(gca, [0 1 0; 0 0 1]);
    colorbar('Ticks', [0.25, 0.75], 'TickLabels', {'Clean Salt', 'Sunscreen'});
    title(sprintf('Detection Result\n%.1f%% Contaminated', sunscreen_pct), ...
        'FontSize', 12, 'FontWeight', 'bold');
    axis image;
    
    fprintf('\n=== NEW SAMPLE RESULTS ===\n');
    fprintf('Sunscreen contamination: %.2f%% of image area\n', sunscreen_pct);
    fprintf('Clean salt: %.2f%% of image area\n', 100 - sunscreen_pct);
    fprintf('Detection completed successfully.\n');
end