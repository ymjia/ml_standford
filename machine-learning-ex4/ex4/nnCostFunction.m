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
X = [ones(m, 1), X];
h_theta = sigmoid([ones(m, 1), sigmoid(X * Theta1')] * Theta2');
## total_cost = 0;
## for i = 1:m
##   yi = zeros(num_labels, 1);
##   yi(y(i)) = 1;
##   hi = h_theta(i, :);
##   cost = log(hi) * yi + log(1 - hi) * (1 - yi);
##   total_cost = total_cost + cost;
##   if (cost > 1)
## 	disp(cost)
##   endif
##   if (mod(i, 50) == 0)
## 	disp(yi)
## 	disp(hi)
## 	kbhit;
##   endif
## end
## total_cost
Y = zeros(m, num_labels);
for i = 1:m
  Y(i, y(i)) = 1;
end

## KENG: 1st row in h_theta * 1st col in Y, NO NEED to multiply 2nd row in h_theta with 1st col in Y
##       so is not a matrix multiply operation. only diagonal elements in h_theta * Y is needed

J = -1 * sum(sum(log(h_theta) .* Y + log(1 - h_theta) .* (1 - Y))) / m; % *Y complete sigma(i:K) sum
regular_item = lambda * (sum(sum(Theta1(:, 2:size(Theta1, 2)).^2)) + sum(sum(Theta2(:, 2:size(Theta2, 2)).^ 2))) / (2 * m);
J = J + regular_item;
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
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for i = 1:m
  yi = zeros(1, num_labels);
  yi(y(i)) = 1;
  xi = X(i, :);
  z2 = xi * Theta1';
  a2 = [1, sigmoid(z2)];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  delta_3 = a3 - yi;
  delta_2 = (delta_3 * Theta2)(2:end) .* sigmoidGradient(z2);
  D2 += delta_3' * a2;
  D1 += delta_2' * xi;
  
end
Theta1_grad = D1 / m;
Theta2_grad = D2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

reg_1 = [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)] * lambda / m;
reg_2 = [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)] * lambda / m;

Theta1_grad += reg_1;
Theta2_grad += reg_2;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
