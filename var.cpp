#include "neuron.h"

std::vector <std::vector <double>> nn::var::dereference_reference_vec (const std::vector <nn::var*> &x) {
	std::vector <std::vector <double>> result;
	for (auto &i : x) {
		assert (i != nullptr);
		result.push_back (i -> v);
	}
	return result;
}

nn::var::var (const std::vector <double> &x) {
	v = x;
	operation = nullptr;
}

nn::var::iterator::iterator () {
	reference = nullptr;
}

nn::var::iterator::iterator (nn::var *ptr) {
	reference = ptr;
}

nn::var::iterator nn::var::iterator::operator = (nn::var *ptr) {
	reference = ptr;
	return *this;
}

bool nn::var::iterator::operator == (nn::var *ptr) {
	return ptr == reference;
}

bool nn::var::iterator::operator == (const nn::var::iterator &i) {
	return reference == i.reference;
}

bool nn::var::iterator::operator != (nn::var *ptr) {
	return ptr != reference;
}

bool nn::var::iterator::operator != (const nn::var::iterator &i) {
	return reference != i.reference;
}

