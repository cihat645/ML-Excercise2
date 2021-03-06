function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%Note: this cost function is designed for a 3 layer neural network
% with any dataset size and any number of labels (classes).


%CREATING Y matrix

Y = eye(num_labels)(y,:);

X = [ones(m,1) X]; %adding column of 1's to matrix X

%%vectorized forward propagation:
a1 = X;
z2 = Theta1 * a1'; %25 x 5000

a2 = sigmoid(z2); %activation function for layer 2
a2 = a2'; 
a2 = [ones(size(a2,1),1) a2]; %adding column of 1's to a2....now has dimensions of 5000 x 26 

hyp = sigmoid(a2 * Theta2'); %hyp has dimensions: 5000 x 10

J = sum(sum((-Y .* log(hyp) - ((1 - Y) .* log(1 - hyp))))) / m;

%sum(sum()) adds up every element in the entire matrix

regularized = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)); %we have the (:, 2:end) cause we do not want to regularize the bias units
J = J + lambda / (2*m) * regularized;


% -------------------------------------------------------------
% Backpropagation (VECTORIZED METHOD):

%STEP 1: CALCULATE THE HYPOTHESIS VIA FORWARD PROPAGATION
a_1 = X; %dimen: 5000 x 401
z_2 = a_1 * Theta1'; % z_2 dimen: (5000 x 25)

a_2 = sigmoid(z_2);
%size(ones(size(X(1,:)))) 

%a_2 = [ones(size(a_2(1,:))) ; a_2]; %adding bias unit (26 x 5000)  ///this was for when it had dimensions 26 x 5000,

a_2 = [ones(size(a_2(:,1)),1) a_2]; %a_2 dimensions: 5000 x 26

z_3 = a_2 * Theta2';
a3 = sigmoid(z_3); %(5000 x 10)

%STEP 2: CALCULATE ERROR IN OUTPUT LAYER

delta3 = a3 - Y; %dimensions: 5000 x 10

%STEP 3: CALCULATE ERROR IN HIDDEN LAYERS 
delta2 = delta3 * Theta2(:, 2:end) .* sigmoidGradient(z_2); %product is 5000 x 25 (:, 2:end) excludes the bias unit
%delta2 dimensions: 5000 x 25

Delta1 = delta2' * a1; %dimensions: 25 x 401
Delta2 = delta3' * a_2; %dimensions: 10 x 26 

Theta1_grad = Delta1 / m; %unregularized gradients
Theta2_grad = Delta2 / m;


%REGULARIZING GRADIENTS:

Theta1(:,1) = 0; %setting first columns = 0
Theta2(:,1) = 0;

Theta1 = Theta1 * ( lambda / m);
Theta2 = Theta2 * (lambda / m);

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;



fprintf('COST = %d\n', J);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
