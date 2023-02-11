function predict(weights, X)
  y_pred = weights[1] + weights[2:end]' * X
  return y_pred
end

function mean_squared_error(weights, X, y)
  m = length(y)
  y_pred = predict(weights, X)
  J = sum((y_pred - y).^2) / (2 * m)
  return J
end

function gradient(weights, X, y)
  m = length(y)
  y_pred = predict(weights, X)
  dJ = X * (y_pred - y) / m
  return dJ
end

function train(weights, X, y, learning_rate, num_iterations)
  m = length(y)
  for i in 1:num_iterations
    dJ = gradient(weights, X, y)
    weights = weights - learning_rate * dJ
  end
  return weights
end

function evaluate(weights, X, y)
  y_pred = predict(weights, X)
  error = mean((y_pred - y).^2)
  return error
end
