%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1); %stores the number of training examples


% Randomly select 100 data points to display
rand_indices = randperm(m);  %the randperm function returns a vector with a random permutation of the numbers between 1 - m
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);



fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

printf('sample values of y:\n');
y(1:10,:)


%What this lab does is take the training data and trains 10 logistic regression
%classifiers, which returns them in the form of weights. The matrix called 'all_theta' 
%stores each of these classifiers 


%A visual representation of the 'network' we're creating here is this:

% 401 units -> 10 units

%This is essentially a neural network without the hidden layer. The parameters we use to map
% the input data to the output data are the synapses between each input unit (pixel) and each output
% unit (logistic regression classifier). Each output unit has been specifically trained to recognize
% the combination of input units that matches the target label (that corresponds to that output unit). 
% For instance, the parameters in row 1 of 'all_theta' have been 'trained' (by minimizing a cost function)
% to "recognize" a pattern of the input pixels that represents the digit 1. The parameters are used to 
% map the input to the output. If the input doesn't match the pattern to which a logistic regression 
% classifier was trained, the output for that logistic regression classifier (LRC) will be lower. 
% However if the pattern of inputs is similar to what the LRC has been trained on, the output will be much 
% higher. These output values represent the model's "confidence" in which class the input belongs. Hence,
% why we choose the class with the highest value in the predictOneVsAll function.








