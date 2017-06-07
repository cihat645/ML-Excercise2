function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X * theta; %vectorized calculation of hypothesis function
error = h - y; % subtracting the hypothesis vector (m x 1) - training values vector (m x 1)
error_squared = error.^2;% using the element-wise operator
J =  1 / (2*m) * sum(error_squared);

% =========================================================================

end
