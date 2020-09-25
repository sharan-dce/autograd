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
