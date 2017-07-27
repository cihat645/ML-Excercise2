function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


%this is my first implementation of how to filter through values and plot them
%m = length(y);
%
%for i = 1:m
%
% if(y(i) == 0),
%   plot(X(i,1), X(i,2), 'ko', 'MarkerFaceColor', 'r');
%
%  else
%   plot(X(i,1), X(i,2), 'g+', 'LineWidth', 2, 'MarkerSize', 7);
%  end
%end



%the reason why I'm using this implementation instead of the first one is because the one above does
% causes issues with the figure legend 

% Find Indices of Positive and Negative Examples
pos = find(y==1); neg = find(y == 0);

%% Plot Examples
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'r', ...
     'MarkerSize', 6);
     
plot(X(pos, 1), X(pos, 2), 'g+','LineWidth', 2, ...
     'MarkerSize', 7);




% =========================================================================



hold off;

end
