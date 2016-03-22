function [ filtermasks, ap_filtermasks ] = get_full_comparisons( params )
%GET_FULL_COMPARISONS Summary of this function goes here
%   Detailed explanation goes here

% VERSION DISAMBIGUATION NOTE: This is the basic version of the function

% pull out useful information from params
orientations = params.filt.orientations;
stdev_pixels = params.filt.stdev_pixels;

% create cell arrays to put things into
filtermasks = cell(length(orientations), length(stdev_pixels), length(orientations), length(stdev_pixels));
ap_filtermasks = cell(length(orientations), length(stdev_pixels), length(orientations), length(stdev_pixels));

h = waitbar(0, 'Total progress');
compcounttotal = length(orientations) * length(stdev_pixels) * length(orientations) * length(stdev_pixels);
compcount = 1;
for o = 1 : length(orientations)
    for f = 1 : length(stdev_pixels)
        filter2 = dogEx(params.filt.y, params.filt.x, stdev_pixels(f) * params.filt.stretchWidth, ...
            stdev_pixels(f), params.filt.negwidth, params.filt.neglen, orientations(o) * pi/180, params.filt.centerW);
        filter2 = trim_filt(filter2, max(abs(filter2(:))) * .01); % trim off edges less than 99% of the max value (trim a lot, but keep the most important bits)
        
        for o2 = 1 : length(orientations)
            for f2 = 1 : length(stdev_pixels)
                waitbar(compcount / compcounttotal, h)
                filter1 = dogEx(params.filt.y, params.filt.x, stdev_pixels(f2) * params.filt.stretchWidth, ...
                    stdev_pixels(f2), params.filt.negwidth, params.filt.neglen, orientations(o2) * pi/180, params.filt.centerW);
                filter1 = trim_filt(filter1, max(abs(filter1(:))) * .01); % trim off edges less than 99% of the max value (trim a lot, but keep the most important bits)
                
                [ filtcomparison, antiphase_filtcomparison ] = compare_filters_alternate( filter1, filter2 );
                
                filtermasks{o,f,o2,f2} = filtcomparison;
                ap_filtermasks{o,f,o2,f2} = antiphase_filtcomparison;
                
                compcount = compcount + 1;
            end
        end
    end
end
close(h)
end

