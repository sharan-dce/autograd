#include <unordered_map>
#include <cmath>
#include <queue>
#include <cassert>
#include <vector>

namespace nn {

	class graph;
	class var;

	struct op {
		virtual std::vector <double> operator () (const std::vector <std::vector <double>>&) = 0;
		virtual std::vector <std::vector <double>> grad (const std::vector <double>&) = 0;
	};

	class add : public op {

		int fan_in;

		void add_to_vector (std::vector <double> &a, const std::vector <double> &b) {
			assert (a.size() == b.size());
			for (size_t i = 0; i < a.size(); i++)
				a[i] += b[i];
		}

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

		void hadamard (std::vector <double> &x, const std::vector <double> &y) {
			for (size_t i = 0; i < x.size(); i++)
				x[i] *= y[i];
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

	class var {
		friend class graph;

		std::vector <double> v;
		op *operation;
		std::vector <var*> inputs;

		std::vector <std::vector <double>> dereference_reference_vec (const std::vector <var*> &x) {
			std::vector <std::vector <double>> result;
			for (auto &i : x) {
				assert (i != nullptr);
				result.push_back (i -> v);
			}
			return result;
		}
		
		public:

		class iterator {
			friend class std::hash <nn::var::iterator>;
			friend class graph;
			friend class var;
			var *reference;
			public:
			iterator () {
				reference = nullptr;
			}
			iterator (var *ptr) {
				reference = ptr;
			}
			iterator operator = (var *ptr) {
				reference = ptr;
				return *this;
			}
			bool operator == (var *ptr) {
				return ptr == reference;
			}
			bool operator == (const iterator &i) {
				return reference == i.reference;
			}
			bool operator != (var *ptr) {
				return ptr != reference;
			}
			bool operator != (const iterator &i) {
				return reference != i.reference;
			}
		};

		var (std::vector <double> x) {
			v = x;
			operation = nullptr;
		}

		template <typename T>
		var (const std::vector <var::iterator> &incoming, T *t_operation) {
			std::vector <var*> raw_in;
			for (auto i : incoming)
				raw_in.push_back (i.reference);
			operation = t_operation;
			inputs = raw_in;
			v = t_operation -> operator () (dereference_reference_vec (raw_in));
		}

	};

	class graph {

		std::vector <var*> var_list;
		std::vector <op*> op_list;

		std::unordered_map <var*, int> find_outdegrees (var *target) {
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

		void add_to_vector (std::vector <double> &a, const std::vector <double> &b) {
			if (a.size () == 0)
				a.resize (b.size());
			assert (a.size () == b.size ());
			for (size_t i = 0; i < a.size(); i++)
				a[i] += b[i];
		}

		public:

		void clear () {
			for (auto i : op_list)
				delete i;
			for (auto i : var_list)
				delete i;
			op_list.clear();
			var_list.clear();
		}

		template <typename T>
		var::iterator add_op (const std::vector <var::iterator> &inputs, const T& t) {
			op_list.push_back (new T);
			var_list.push_back (new var (inputs, *op_list.rbegin ()));
			return var::iterator (*var_list.rbegin());
		}

		std::vector <std::vector <double>> compute_gradients (var::iterator tar, const std::vector <var::iterator> &var_list) {
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

		~graph () {
			clear();
		}

	};

}

namespace std {
	template <>
	struct hash <nn::var::iterator> {
		hash <nn::var*> reference_hash;
		std::size_t operator () (const nn::var::iterator &it) const {
			return reference_hash (it.reference);
		}
	};

}
