import pandas as pd # To read CSV data
from matplotlib import pyplot as plt # To graph and display data


def main():
	# Load data
	df = pd.read_csv("weight-height.csv")

	# Seperate data into x and y values
	# Weights
	x = list(df['Weight'])

	# Heights
	y = list(df['Height'])

	#########################
	# Create graph of data
	plt.scatter(x, y)

	plt.xlabel("Weight")
	plt.ylabel("Height")
	plt.show()


	# Initial guess at the parameters
	params = {'w': 1.2, 'b': 30}
	#########################

	#########################
	# Graph data and untrained predictions
	y_hat = forward_pass(x, params)
	cost = get_cost(y, y_hat)

	plt.scatter(x, y)
	plt.scatter(x, y_hat)

	plt.show()
	#########################

	#########################
	# Graph data and trained predictions
	params, _ = train(x[:-100], y[:-100], params, epochs = 2000)

	y_hat = forward_pass(x[-100:], params)

	plt.scatter(x[-100:], y[-100:])
	plt.scatter(x[-100:], y_hat[-100:])

	plt.show()
	#########################


	#########################
	# Loop to test model

	while True:

		user_input = input('Input the wieght of a person in' +
			' pounds and I will predict his/her height. Enter' + 
			' "EXIT" and this program will end.\n>> ')

		if user_input == 'EXIT':
			exit()

		try:
			weight = float(user_input)

			prediction = forward_pass([weight], params)

			print(f"The corresponding person is {prediction} inches tall.")

		except ValueError:

			print('Invalid input. Please try again')
	#########################



def train(x: list, y: list, params: dict, epochs: int = 1,
	learning_rate: float = 0.00003, vocal = True) -> tuple:
	'''
	Trains parameters to a linear model (i.e. y = w * x + b).

	:param x: The input data.
	:param y: The output data.
	:param params: A dictionary which contains the parameters of the linear model:
		'w' - the weight for inputs 'x'
		'b' - the bias
	:param epochs: The number of training itterations.
	:param learning_rate: The number which multiply's the gradiants to update parameters.
		If the 'learning_rate' is too large, the model will not train properly.
	:param vocal: If true, the function prints the current cost of the model to the standard
		output every 100 epochs.
	:returns: (new_params, cost_history) where 'new_params' is the new and trained
		set of parameters and 'cost_history' is a list containing the cost
		after each epoch.
	'''

	# Copy params dictionary to avoid side effects.
	new_params = params.copy()

	cost_history = []

	# Update paramaters 'epochs' times
	for i in range(epochs):

		y_hat = forward_pass(x, new_params)
		cost = get_cost(y, y_hat)
		grads = backward_pass(x, y, new_params)

		# Print cost status
		if vocal and i % 100 == 0:
			print(cost)

		# Unpack gradients
		dw = grads['dw']
		db = grads['db']
	
		# Update weight and bias
		new_params['w'] -= dw * learning_rate
		new_params['b'] -= db * learning_rate

		cost_history.append(cost)

	return new_params, cost_history



def forward_pass(x: list, params: dict) -> list:
	'''
	Calculates one forward pass for a linear model.
		For example, if 'w' = 2 and 'b' = 0, then an input
		of [1,2,3] would result in an output of [2,4,6].

	:param x: The input data. Should be a 1D list
	:param params: A dictionary which contains the parameters of the linear model:
		'w' - the weight for inputs 'x'
		'b' - the bias
	:returns: The outputs for inputs 'x'.
	'''

	# Unpack parameters from dictionary
	w = params['w']
	b = params['b']

	# The outputs
	y_hat = []

	# Calculate outputs for each input
	for val in x:
		y_hat.append(val * w + b)

	return y_hat

def get_cost(y: list, y_hat: list) -> float:
	'''
	Calculates the mean squared error for predictions.

	:param y: The actual outputs.
	:param y_pred: The predicted outputs.
	:returns: The cost of the predictions
	'''
	cost = 0.0

	# Sum the costs for each output/prediction pair
	for (y_i, yh_i) in zip(y, y_hat):

		cost += (y_i - yh_i)**2

	# Average the costs
	cost = cost / len(y)

	return cost

def backward_pass(x: list, y: list, params: dict) -> dict:
	'''
	Completes one step of backward propagation for a linear model
		using MSE as the cost function. 

	:param x: The input data.
	:param y: The output data.
	:param params: A dictionary which contains the parameters of the linear model:
		'w' - the weight for inputs 'x'
		'b' - the bias
	:returns: A dictionary containing the gradient for each paramater:
		'dw' - The derivative of the cost function with respect to 'w' (dC/dw)
		'db' - The derivative of the cost function with respect to 'b' (dC/db)
	'''

	# Unpack parameters from dictionary
	w = params['w']
	b = params['b']

	# dC/dw
	dw = 0.0

	# dC/db
	db = 0.0

	# Sum gradients for each input/output pair
	for x_i, y_i in zip(x, y):

		# Both of these formula are the formula for 
		# the derivative of the cost function with respect
		# to 'w' and 'b' respectively
		dw += -2 * x_i *(y_i - (w * x_i + b))

		db += -2 * (y_i - (w * x_i + b))

	# Average the sum of the gradients
	dw = dw / len(x)
	db = db / len(x)

	grads = {'dw': dw, 'db': db}

	return grads

main()