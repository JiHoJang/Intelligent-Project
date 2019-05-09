#ifndef MLarchitecture_h
#define MLarchitecture_h


template<class W>
class Weight;

template<class M>
class Matrix;

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
	maxPoolIdx
};

#endif