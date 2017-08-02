# this function gives the constant c which is needed to decide the savintzky - golay filter window
# window = len(data) / c where c is a constant strongly related to the error contained in the data set


def c(error):

	c = 10.25 + (10676070 / (1 + ((error/1.24)**5.36)))
	return c

x_ex = 10
print(c(x_ex))