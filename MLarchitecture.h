#ifndef MLarchitecture_h
#define MLarchitecture_h


template<class W>
class Weight;

template<class M>
class Matrix;

template <class L>
class Layer;

enum datatype {
	int32,
	int64,
	float32,
	float64,
	read,
	random,
	result,
	input,
	relu,
	sigmoid,
	conv,
	weight,
	maxPooling,
	maxPoolIdx,
	reshape,
	matmul
};

#endif