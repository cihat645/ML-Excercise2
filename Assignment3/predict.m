function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%this prediction function takes in the trained parameters from the neural network and applies the 
%weights to the input data in this function (X) and outputs a prediction

%simple implementation that only works for 3 layer network:

X = [ones(m,1), X]; %add x0 column to matrix X

z2 = Theta1 * X'; %this will return a 25 x 5000 matrix
% we just applied all the parameters for the first layer to the training data, now we must use the activation function
% to get the data ready for the second layers
a2 = sigmoid(z2); %applying activation function

%now we have a2, 25 x 5000, we must add a column of 1's to this to add the bias unit
%before we can add a column, we need to tranpose a2
a2 = a2';
a2 = [ones(size(a2,1),1), a2]; %a2 now has dimensions 5000 x 26
a2 = a2'; %a2 now has dimensions of 26 x 5000

z3 = Theta2 * a2; %now we are applying theta2, the second matrix of parameters to our data
%to prepare it for the third layer of neurons
%now we must apply the activation function to z3 to get our expected probabilities

hyp = sigmoid(z3); 
%hyp is a 10 x 5000 matrix. Every element of this matrix corresponds to the probabilities each example has for being 
%each number (or a member of each class). For instance, the value of element (5,1) in this matrix says:
% "This is the probability that the first training example is the number 5 (or belongs to class 5)"

%Now that we have all the probabilities for each class, we need to find the class with the highest probability for 
%each training example


%transposing hyp because we return it as a column vector
hyp = hyp'; %dimensions now are: 5000 x 10
[probability_value, p] = max(hyp, [], 2); % this returns the index of the class that corresponds to the maximum 
%probability for each row of the matrix 







%our output should be the dimensions of (number_of_rows_in_matrix_X) x (1)



% =========================================================================


end
