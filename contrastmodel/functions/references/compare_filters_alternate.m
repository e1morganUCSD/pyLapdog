function [ filtcomparison, ap_filtcomparison ] = compare_filters_alternate( filter1, filter2 )
%COMPARE_FILTERS Summary of this function goes here
%   Detailed explanation goes here

% filtcomparison = zeros(size(filter1, 1) + size(filter2, 1), ...
%     size(filter1, 2) + size(filter2, 2));

%ap_filtcomparison = zeros(size(filtcomparison));

filter1_selfcomp = sum(sum(filter1 .* filter1));
filter2_selfcomp = sum(sum(filter2 .* filter2));
comparison_denom = sqrt(filter1_selfcomp * filter2_selfcomp);

% antiphase_filter1 = -1 .* filter1;
% antiphase_filter1_selfcomp = sum(sum(antiphase_filter1 .* antiphase_filter1));
% antiphase_comparison_denom = sqrt(antiphase_filter1_selfcomp * filter2_selfcomp);

% pad filter 2 to be at least the same size as filter1
filter2 = pad_filter2(filter2, size(filter1, 1), size(filter1, 1));
% filter2 = [zeros(size(filter2, 1), size(filter1, 2)), ...
%     filter2, zeros(size(filter2, 1), size(filter1, 2))];
% 
% filter2 = [zeros(size(filter1, 1), size(filter2, 2)); filter2; ...
%     zeros(size(filter1, 1), size(filter2, 2))];

% % create a blank slate to put filter 1 on and move it around
% slate = zeros(size(filter2));

%filter1r = reshape(filter1, 1, []);

% h = waitbar(0, 'Processing...');

func = @(x) sum(filter1(:) .* x(:));

%filtcomparison = colfilt(filter2, size(filter1), [40,40], 'sliding', func);
filtcomparison = nlfilter(filter2, size(filter1), func);

filtcomparison = filtcomparison ./ comparison_denom;

ap_filtcomparison = filtcomparison .* -1;

% for x = 1:size(slate, 2) - size(filter1, 2)
%     waitbar(x / (size(slate, 2) - size(filter1, 2)), h);
%     for y = 1:size(slate, 1) - size(filter1, 1)
%         temp = slate;
%         temp(y:y+size(filter1,1)-1, x:x+size(filter1,2)-1) = filter1;
%         filtcomparison(y, x) = sum(sum(filter2 .* temp)) / comparison_denom;
%         antiphase_filtcomparison(y, x) = filtcomparison(y, x) * -1;
%     end
% end

% close(h)
end

