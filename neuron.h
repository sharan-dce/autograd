#include <unordered_map>
#include <cmath>
#include <queue>
#include <cassert>
#include <vector>

#ifndef NEURON
#define NEURON

namespace nn {

	class graph;
	class var;

	struct op {
		virtual std::vector <double> operator () (const std::vector <std::vector <double>>&) = 0;
		virtual std::vector <std::vector <double>> grad (const std::vector <double>&) = 0;
	};

	
	class var {

		public:
		friend class graph;
		class iterator;

		private:

		std::vector <double> v;
		op *operation;
		std::vector <var*> inputs;

		std::vector <std::vector <double>> dereference_reference_vec (const std::vector <var*> &);

		template <typename T>
		var (const std::vector <var::iterator> &incoming, T *t_operation) {
			std::vector <var*> raw_in;
			for (auto i : incoming)
				raw_in.push_back (i.reference);
			operation = t_operation;
			inputs = raw_in;
			v = t_operation -> operator () (dereference_reference_vec (raw_in));
		}
		
		public:

		class iterator {

			friend class std::hash <nn::var::iterator>;
			friend class graph;
			friend class var;
			var *reference;

			public:

			iterator ();
			iterator (var *);
			iterator operator = (var *);
			bool operator == (var *);
			bool operator == (const iterator &);
			bool operator != (var *);
			bool operator != (const iterator &);
		};

		var (const std::vector <double> &);

	};

	class graph {

		std::vector <var*> var_list;
		std::vector <op*> op_list;

		std::unordered_map <var*, int> find_outdegrees (var *);

		void add_to_vector (std::vector <double> &, const std::vector <double> &);

		public:

		void clear ();

		template <typename T>
		var::iterator add_op (const std::vector <var::iterator> &inputs, const T& t) {
			op_list.push_back (new T);
			var_list.push_back (new var (inputs, *op_list.rbegin ()));
			return var::iterator (*var_list.rbegin());
		}

		std::vector <std::vector <double>> compute_gradients (var::iterator tar, const std::vector <var::iterator> &var_list);

		~graph ();
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

#endif
