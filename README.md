# Autograd

Autograd is an autodifferentiation library for C++. There are a few in-built operators that could be used to specify a wide range of computation graphs, and users can define custom operator classes as well, by simply inheriting from ```nn::op```.
```autograd```'s API should be used to specify graphs by thinking about operators as modules that take in a number of vectors and produce an output vector.

### Examples
1. Using the classes
```
#include <iostream>
#include "autograd.h"
#include "nnops.h"

void print_vector (const std::vector <double> &x) {
	for (auto i : x)
		std::cout << i << ' ';
	std::cout << std::endl;
}

int main () {
	nn::graph g;
	nn::var x ({0.5, -0.1, 0.012, 0.00122, -0.92});
	nn::var y ({-0.1, -0.019, -0.0965, 0.0127});
	auto x_exp = g ({&x}, nn::exp ());
	auto output = g ({x_exp, &y}, nn::concat ());
	output = g ({g ({output}, nn::tanh ())}, nn::reduce_sum ());
	output = g ({output}, nn::sigmoid ());
	output = g ({output}, nn::prod (0.5));
	auto gr = g.compute_gradients (output, {&x, &y});
	for (auto &i : gr)
		print_vector (i);
}
```

Output:
```
0.00370426 0.00714669 0.00681578 0.00685549 0.00557598 
0.0161718 0.0163281 0.0161829 0.0163314
```

2. Creating a custom op
```
class add : public nn::op {

	int fan_in;
	
	public:

	std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
		assert (input.size() > 0);
		size_t vector_size = input[0].size();
		std::vector <double> result (vector_size);
		for (const auto &i : input) {
			assert (i.size() == vector_size);
			add_to_vector (result, i);
		}
		fan_in = input.size();
		return result;
	}

	std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
		return std::vector <std::vector <double>> (fan_in, output_grad);
	}

};
```
### Guide

  - autograd is defined inside the namespace ```nn```
  - There are 3 main classes: ```nn::graph```, ```nn::var``` and ```nn::op``` (pure virtual)

##### ```nn::var```
This represents the library's variable class.
  - ```nn::var::var (std::vector <double>)``` is the only public constructor, and takes in a double typed vector of values to initialize the object.
  - ```nn::var::get_value ()``` returns a vector with the value of the ```var``` object.
  - ```var``` objects cannot be modified, and are either standalone as specified above (can be reconstructed if new values are to be used), or dependent nodes in the graph, which can be constructed as specified in the ```nn::graph``` section below.
  - ```nn::var::iterator``` is a smart pointer class which will come in handy while creating nodes in the graph

##### ```nn::graph```
Objects of this class represent and store computation graphs. All you need to do is simply create an object, and start adding dependent nodes to it.
  - ```nn::var::iterator nn::graph::operator () (const std::vector <nn::var::iterator> &inputs, const T &t)```: on calling the graph object with the above arguments (inputs node iterator vector and an object of the operator to be used - can be a temporary instance), the graph adds a node and returns a ```nn::var::iterator``` pointing to it.
  - ```void nn::graph::clear ()``` clears the graph of all dependent nodes (all dependent nodes that **belong to the graph**)
  - ```std::vector <std::vector <double>> nn::graph::compute_gradients (nn::var::iterator tar, const std::vector <var::iterator> &var_list)``` can be used to compute the gradients of required ```var```s by providing a list of iterators (or pointers) pointing to them. ```tar``` represents the variable with respect to which gradients are to be computed, and has to be a scalar (which is a ```var``` having a single dimensional array value). 

##### ```nn::op```
This is a pure virtual class, and cannot be instantiated. All operators (as defined in ```nnops.h```) inherit from this class (you can write a custom operator class this way), which enforces the definition of the following 2 functions:
  - ```		virtual std::vector <double> operator () (const std::vector <std::vector <double>>&)``` - this makes an op child class object a callable - this defines the operation. Expects a vector of vectors to operate over, and returns a vector.
  - ```virtual std::vector <std::vector <double>> grad (const std::vector <double>&) = 0``` - this method defines the derivatives needed to be passed back during backpropagation. This is called after the () operator during backpropagation, and is given the gradient of the operator's output, hence consider the object to be stateful, and save whatever is required during an object () call. This method is expected to return a vector of vectors - each vector, in order being the gradient vector of the inputs to the operator. 

##### Provided operators
Implementations of these can be found in ```nnops.h```. 
1) Softmax - ```softmax```
2) Dot Product - ```dot```
3) Product - ```prod``` - when constructed as ```prod (double scale)```, this operator scales the input vector by the constant ```scale```, otherwise finds the element-wise product of the incoming vectors
4) Reduce sum - ```reduce_sum```
5) Add - ```add```
6) Subtract - ```subtract```
7) Tanh - ```tanh```
8) Relu - ```relu```
9) Exp - ```exp```
10) Log - ```log```
11) Sigmoid - ```sigmoid```
12) Concat - ```concat```
13) Power - ```power```
### Requirements

  - \>= C++14

