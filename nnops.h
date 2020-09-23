#include "neuron.h"

// operations
namespace nn {
	void add_to_vector (std::vector <double> &x, const std::vector <double> &y) {
		assert (x.size() == y.size());
		for (size_t i = 0; i < x.size(); i++)
			x[i] += y[i];
	}
	void subtract_from_vector (std::vector <double> &a, const std::vector <double> &b) {
		assert (a.size() == b.size());
		for (size_t i = 0; i < a.size(); i++)
			a[i] -= b[i];
	}
	void hadamard (std::vector <double> &x, const std::vector <double> &y) {
		assert (x.size () == y.size ());
		for (size_t i = 0; i < x.size(); i++)
			x[i] *= y[i];
	}



	class add : public op {

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

	class exp : public op {

		std::vector <double> computed_output;

		void exp_vector (std::vector <double> &x) {
			for (auto &i : x)
				i = std::exp (i);
		}

		
		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 1);
			computed_output = input[0];
			exp_vector (computed_output);
			return computed_output;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			std::vector<std::vector<double>> input_grads (1, computed_output);
			hadamard (input_grads[0], output_grad);

			return input_grads;
		}

	};

	class subtract : public op {

		void negate (std::vector <double> &x) {
			for (double &i : x)
				i = -i;
		}

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 2);
			std::vector <double> result = input[0];
			subtract_from_vector (result, input[1]);
			return result;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			std::vector <std::vector <double>> input_grads (2, output_grad);
			negate (input_grads[1]);
			return input_grads;
		}

	};

	class prod : public op {

		std::vector <std::vector <double>> input_cache;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 2 and input[0].size() == input[1].size());
			std::vector <double> result = input[0];

			hadamard (result, input[1]);

			input_cache = input;

			return result;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size() == input_cache[0].size());
			
			swap (input_cache[0], input_cache[1]);

			hadamard (input_cache[0], output_grad);
			hadamard (input_cache[1], output_grad);

			return input_cache;
		}

	};

	class reduce_sum : public op {

		int dimensions;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 1);
			std::vector <double> output (1);
			dimensions = input[0].size();
			for (double i : input[0])
				output[0] += i;
			return output;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size() == 1);
			return {std::vector <double> (dimensions, output_grad[0])};
		}
	};

	class dot : public op {

		std::vector <std::vector <double>> cached_input;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 2);
			assert (input[0].size () == input[1].size ());

			cached_input = input;
			
			std::vector <double> result = input[0];
			for (size_t i = 0; i < input[0].size (); i++)
				result[i] *= input[1][i];
			return result;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size () == 1);
			swap (cached_input[0], cached_input[1]);
			std::vector <double> scale_value (cached_input[0].size (), output_grad[0]);
			hadamard (cached_input[0], scale_value);
			hadamard (cached_input[1], scale_value);
			return cached_input;
		}
	};

	class relu : public op {

		std::vector <std::vector <double>> cached_input;
		std::vector <bool> mask;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 1);

			mask.clear ();
			std::vector <double> result = input[0];
			for (double &i : result) {
				i = std::max (i, 0.0);
				mask.push_back ((i > 0));
			}
			return result;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			std::vector <std::vector <double>> result (1, output_grad);
			for (size_t i = 0; i < output_grad.size (); i++)
				result[0][i] *= mask[i];
			mask.clear ();
			return result;
		}
	};
}


