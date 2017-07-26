function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); %creates a vector to store the cost for each iteration
%I use J_history to ensure that gradient descent is functioning correctly


    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  
    for iter = 1:num_iters

    hyp = X * theta;
    diff = hyp - y;
    theta_change = (alpha / m) * (X' * diff);  % no need to use sum() because the vector multiplication sums them up
    theta = theta - theta_change;
    J_history(iter) = computeCost(X,y, theta) % printing value of J(theta) to ensure it is decreasing 
    % Saving the cost J in every iteration    

end


end
