function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); %initialize the gradient vector

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%



hyp = sigmoid(X * theta); %returns an (m x 1) vector, X has dimensions (m x n+1), theta has dimensions (n+1, 1)

cost = (-y' * log(hyp) - (1 - y)' * log(1-hyp)); %vectorized cost function used for logistic regression

J = (1 / m) * cost; %notice, no need for sum() because we are using the vectorized implementation of the cost function for logistic regression

grad = (1 / m) * X' * (hyp - y);






% =============================================================

end
