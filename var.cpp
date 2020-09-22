#include "neuron.h"

std::vector <std::vector <double>> nn::dereference_reference_vec (const std::vector <nn::var*> &x) {
			std::vector <std::vector <double>> result;
			for (auto &i : x) {
				assert (i != nullptr);
				result.push_back (i -> v);
			}
			return result;
		}


