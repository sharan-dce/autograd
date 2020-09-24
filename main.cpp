#include <iostream>
#include "neuron.h"
#include "nnops.h"

void publish (const std::vector <double> &x) {
	for (auto i : x)
		std::cout << i << ' ';
	std::cout << std::endl;
}

int main () {
	nn::graph g;
	nn::var x ({0.5, -0.1, 0.012, 0.00122, -0.92});
	nn::var y ({-0.1, -0.019, -0.0965, 0.0127});
	auto x_exp = g.add_op <nn::exp> ({&x});
	auto output = g.add_op <nn::concat> ({x_exp, &y});
	publish (output -> get_value ());
	output = g.add_op <nn::reduce_sum> ({g.add_op <nn::tanh> ({output})});
	output = g.add_op <nn::sigmoid> ({output});
	auto gr = g.compute_gradients (output, {&x, &y});
	for (auto &i : gr)
		publish (i);
}
