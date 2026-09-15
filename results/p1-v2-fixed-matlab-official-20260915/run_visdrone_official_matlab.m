function run_visdrone_official_matlab(toolkit_dir, dataset_dir, detections_dir, output_json)
% Run the original VisDrone DET utilities on the same sorted DET TXT files.
% Requires a working MATLAB license. Does not alter official evaluator code.
addpath(fullfile(toolkit_dir, 'utils'));
gt_dir = fullfile(dataset_dir, 'annotations');
img_dir = fullfile(dataset_dir, 'images');
names = findImageList(gt_dir);
count = length(names);
assert(count == 548, 'Expected the complete 548-image validation split.');
[gt, det] = saveAnnoRes(gt_dir, detections_dir, img_dir, count, names);
[ap, ap50, ap75, ar1, ar10, ar100, ar500] = calcAccuracy(count, gt, det);
result = struct('evaluator', 'original VisDrone DET MATLAB utilities', ...
    'images', count, 'AP', ap, 'AP50', ap50, 'AP75', ap75, ...
    'AR1', ar1, 'AR10', ar10, 'AR100', ar100, 'AR500', ar500, ...
    'toolkit_dir', toolkit_dir, 'detections_dir', detections_dir);
file = fopen(output_json, 'w', 'n', 'UTF-8');
assert(file ~= -1, 'Cannot open output JSON; create its parent directory first.');
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', jsonencode(result, PrettyPrint=true));
disp(result);
end
