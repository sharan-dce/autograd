#include "neuron.h"
#include <iostream>

// operations
namespace nn {
	const double EPSILON = 1e-8;
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
	double dot_product (const std::vector <double> &x, const std::vector <double> &y) {
		assert (x.size () == y.size ());
		double dot = 0;
		for (size_t i = 0; i < x.size (); i++)
			dot += x[i] * y[i];
		return dot;
	}
	void scale_vector (std::vector <double> &x, double f) {
		for (auto &i : x)
			i *= f;
	}
	void sigmoid_vector (std::vector <double> &x) {
		for (double &i : x)
			i = 1.0 / (1.0 + std::exp (-i));
	}
	void tanh_vector (std::vector <double> &x) {
		for (double &i : x) {
			double exp = std::exp (i * 2);
			i = (exp - 1) / (exp + 1);
		}
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
		bool scaled;
		double scale;

		std::vector<double> prod_call (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 2 and input[0].size() == input[1].size());
			std::vector <double> result = input[0];

			hadamard (result, input[1]);

			input_cache = input;

			return result;
		}

		std::vector<std::vector<double>> prod_grad (const std::vector<double> &output_grad) {
			assert (output_grad.size() == input_cache[0].size());
			
			swap (input_cache[0], input_cache[1]);

			hadamard (input_cache[0], output_grad);
			hadamard (input_cache[1], output_grad);

			return input_cache;
		}

		std::vector<double> scale_call (const std::vector<std::vector<double>> &input) {
			assert (input.size() == 1);
			std::vector <double> result = input[0];

			for (auto &i : result)
				i *= scale;

			return result;
		}

		std::vector<std::vector<double>> scale_grad (const std::vector<double> &output_grad) {
			std::vector <double> result = output_grad;
			for (auto &i : result)
				i *= scale;
			return {result};
		}


		public:

		prod () {
			scaled = false;
		}

		prod (double s) {
			scaled = true;
			scale = s;
		}

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			if (scaled)
				return scale_call (input);
			else
				return prod_call (input);
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			if (scaled)
				return scale_grad (output_grad);
			else
				return prod_grad (output_grad);
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
			
			auto result = dot_product (input[0], input[1]);
			return {result};
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

	class concat : public op {

		std::vector <int> sizes;
		size_t sizes_sum;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			sizes_sum = 0;
			for (const std::vector <double> &i : input)
				sizes.push_back (i.size ()), sizes_sum += i.size ();
			// std::cout << sizes_sum << std::endl;
			
			std::vector <double> result;

			for (const std::vector <double> &i : input)
				result.insert (result.end (), i.begin (), i.end ());
			return result;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			// std::cout << sizes_sum << ' ' << output_grad.size () << std::endl;
			assert (output_grad.size () == sizes_sum);
			auto it = output_grad.begin ();
			std::vector <std::vector <double>> input_grads;
			for (auto i : sizes)
				input_grads.push_back (std::vector <double> (it, it + i)), it += i;
			return input_grads;
		}
	};

	class softmax : public op {

		std::vector <double> cached_output;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size () == 1);
			double min = *std::min_element (input[0].begin (), input[0].end ());
			cached_output.resize (input[0].size ());
			double scale = 0;
			for (const double &i : input[0])
				scale += std::exp (i - min);
			for (size_t i = 0; i < input[0].size (); i++)
				cached_output[i] = std::exp (input[0][i] - min) / scale;
			return cached_output;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size () == cached_output.size ());
			std::vector <double> t1 = cached_output, t2 = cached_output;
			hadamard (t1, output_grad);
			scale_vector (t2, dot_product (output_grad, t2));
			subtract_from_vector (t1, t2);
			return {t1};
		}
	};

	class sigmoid : public op {

		std::vector <double> cached_output;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size () == 1);
			cached_output = input[0];
			sigmoid_vector (cached_output);
			return cached_output;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size () == cached_output.size ());
			for (size_t i = 0; i < output_grad.size (); i++)
				cached_output[i] *= (1. - cached_output[i]) * output_grad[i];
			return {cached_output};
		}
	};

	class tanh : public op {

		std::vector <double> cached_output;

		public:

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size () == 1);
			cached_output = input[0];
			tanh_vector (cached_output);
			return cached_output;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size () == cached_output.size ());
			for (size_t i = 0; i < output_grad.size (); i++)
				cached_output[i] = (1. - cached_output[i] * cached_output[i]) * output_grad[i];
			return {cached_output};
		}
	};

	class log : public op {

		std::vector <double> cached_input;
		double epsilon;

		public:

		log (double e = EPSILON) {
			epsilon = e;
		}

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size () == 1);
			assert (*std::min_element (input[0].begin (), input[1].begin ()) > 0);
			cached_input = input[0];
			auto output = cached_input;
			for (auto &i : output)
				i = std::log (std::max (i, epsilon));
			return output;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size () == cached_input.size ());
			for (size_t i = 0; i < output_grad.size (); i++)
				cached_input[i] = output_grad[i] / std::max (cached_input[i], epsilon);
			return {cached_input};
		}
	};

	class power : public op {

		double pow;
		std::vector <double> cache;

		public:

		power (double x) {
			pow = x;
		}

		std::vector<double> operator () (const std::vector<std::vector<double>> &input) {
			assert (input.size () == 1);
			cache = input[0];
			for (auto &i : cache)
				i = std::pow (i, pow - 1);
			auto result = cache;
			for (size_t i = 0; i < result.size (); i++)
				result[i] *= input[0][i];
			return result;
		}

		std::vector<std::vector<double>> grad(const std::vector<double> &output_grad) {
			assert (output_grad.size () == cache.size ());
			for (size_t i = 0; i < output_grad.size (); i++)
				cache[i] *= pow * output_grad[i];
			return {cache};
		}
	};
}


