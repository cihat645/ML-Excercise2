function [g, a] = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

%the size() function returns the number of rows and columns

%   size ([1, 2; 3, 4; 5, 6])
%             => [ 3, 2 ]




% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


%g = 1 / (1 + e.^-(z));


%unvectorized implementation
[x, y] = size(z);  %x = # of rows, y = # col 
for i = 1:x,
  for j = 1:y,
     g(i,j) = 1 / (1 + e^-(z(i,j)));
   end
end







% =============================================================

end
