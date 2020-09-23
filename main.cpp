#include <iostream>
#include "neuron.h"
#include "nnops.h"

int main () {
	nn::graph g;
	nn::var x ({0.5});
	nn::var y ({-0.1});
	nn::var::iterator output = g.add_op ({&x, &y}, nn::add());
	output = g.add_op ({output}, nn::exp());
	auto gradients = g.compute_gradients (output, {&x, &y});
	std::cout << gradients[0][0] << ' ' << gradients[1][0] << std::endl;
}
