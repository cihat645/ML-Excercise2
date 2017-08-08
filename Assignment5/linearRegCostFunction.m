function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hyp = X * theta; %X: 12 x 2 theta: 2 x 1 = hyp: 12 x 1 
J = (sum((hyp - y) .^2)) / (2*m);
theta_reg = theta(2:end); %creating a theta_reg term to store all the theta values to be summed (excluding theta 0, which, in octave, is theta(1))
reg = lambda / (2 * m) * (sum(theta_reg .^ 2)); 
J =  J + reg;

grad(1) = (1/m) * sum((hyp - y) .* X(:, 1));
for i = 2:numel(grad),
  grad(i) = ((1/m) * sum((hyp - y) .* X(:, i))) + (lambda / m) * theta(i);
end







% =========================================================================

grad = grad(:);

end
