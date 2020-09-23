#include "neuron.h"

std::unordered_map <nn::var*, int> nn::graph::find_outdegrees (nn::var *target) {
	std::unordered_map <var*, int> outdegrees;
	std::unordered_map <var*, bool> visited;
	outdegrees[target] = 0;
	visited[target] = true;
	std::queue <var*> q;
	q.push (target);

	while (q.size()) {
		var *node = q.front();
		q.pop();

		for (var *i : node -> inputs) {
			outdegrees[i] += 1;
			if (visited[i] == false) {
				visited[i] = true;
				q.push (i);
			}
		}
	}
	return outdegrees;
}

void nn::graph::add_to_vector (std::vector <double> &a, const std::vector <double> &b) {
	if (a.size () == 0)
		a.resize (b.size());
	assert (a.size () == b.size ());
	for (size_t i = 0; i < a.size(); i++)
		a[i] += b[i];
}

void nn::graph::clear () {
	for (auto i : op_list)
		delete i;
	for (auto i : var_list)
		delete i;
	op_list.clear();
	var_list.clear();
}

std::vector <std::vector <double>> nn::graph::compute_gradients (nn::var::iterator tar, const std::vector <nn::var::iterator> &var_list) {
	var *target = tar.reference;
	auto outdegrees = find_outdegrees (target);
	std::unordered_map <var*, std::vector <double>> gradients;
	std::queue <var*> q;

	gradients[target] = {1.0};
	q.push (target);

	while (q.size()) {
		auto node = q.front ();
		q.pop ();

		if (node -> operation == nullptr)
			continue;
		auto input_gradients = node -> operation -> grad (gradients[node]);


		for (size_t i = 0; i < input_gradients.size(); i++) {
			var *in_node = (node -> inputs)[i];
			add_to_vector (gradients[in_node], input_gradients[i]);
			outdegrees[in_node]--;
			if (outdegrees[in_node] == 0)
				q.push (in_node);
		}
	}
	std::vector <std::vector <double>> result;


	for (auto &i : var_list)
		result.push_back (gradients[i.reference]);
	return result;
}

nn::graph::~graph () {
	clear ();
}
