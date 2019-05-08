#include <iostream>
#include <vector>

using namespace std;

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
	pooling,
	poolIdx
};

template<typename F>
F RetValue(int method) {
	F ans;

	return ans;
}

template<class M>
class Matrix {
private:
	M*** mat;
	int method, row, col, channels;
	int type;

public:
	Matrix(int _method, int _row, int _col, int _channels, int _type) {
		method = _method;
		row = _row;
		col = _col;
		channels = _channels;
		type = _type;

		if (method == random) {
			mat = new(M * *)(channels);
			for (int j = 0; j < channels; j++) {
				mat[j] = new(M*)(row);
				for (int k = 0; k < row; k++) {
					mat[j][k] = new (M)(col);
					for (int l = 0; l < col; l++)
						mat[j][k][l] = RetValue<M>(method);
				}
			}
		}
		else {
			mat = new(M * *)(channels);
			for (int j = 0; j < channels; j++) {
				mat[j] = new(M*)(row);
				for (int k = 0; k < row; k++) {
					mat[j][k] = new (M)(col);
				}
			}
		}
	}
	~Matrix() {
		for (int j = 0; j < channels; j++) {
			for (int k = 0; k < row; k++) {
				delete[] mat[j][k];
			}
			delete[] mat[j];
		}
		delete[] mat;
	}
};

template<class W>
class Weight;

template<class L>
class Layer {
private:
	vector<Matrix<L> > mat; // for calculate functions
	int row, col, channels;
	int dim;
	int strides[2];
public:
	Weight* prev = NULL;
	Weight* next = NULL;

	Layer(int method, int dimention[], int type) {
		dim = sizeof(dimention) / 4;
		if (dim == 1) {
			row = dimention[0];
			col = 1;
			channels = 1;
		}
		else if (dim == 2) {
			row = dimention[0];
			col = dimention[1];
			channels = 1;
		}
		else if (dim == 3) {
			row = dimention[0];
			col = dimention[1];
			channels = dimention[2];
		}
		else {
			cout << "error in placeholder\n";
			exit(1);
		}

		Matrix<L> temp(method, row, col, channels, type);

		mat.push_back(temp);
	}

	~Layer() {
		for (auto it = mat.begin(); it != mat.end(); it++) {
			it->~Matrix();
		}
		mat.clear();
	}

	template<class We>
	Matrix<L> conv2d(We w, int strides[2], bool padding) {
		this->strides[0] = strides[0];
		this->strides[1] = strides[1];

		if (padding == false) {
			row -= strides[0] - 1;
			col -= strides[1] - 1;
		}

		Matrix<L> ret;

		return ret;
	}
};

template<class W>
class Weight {
private:
	vector<Matrix<W> > mat; // for next channels
	int row, col, channels;
	int nextChannels;
public:
	Layer* prev = NULL;
	Layer* next = NULL;

	Weight(int method, int kernel[4]) {
		row = kernel[0];
		col = kernel[1];
		channels = kernel[2];
		nextChannels = kernel[3];

		for (int i = 0; i < nextChannels, i++) {
			Matrix<W> temp(method, row, col, channels, weight);
			mat.push_back(temp);
		}
	}

	~Weight() {
		for (auto it = mat.begin(); it != mat.end(); it++) {
			it->~Matrix();
		}
		mat.clear();
	}

	update() {

	}
};


//template <typename P>
//class placeholder {
//private:
//	//void* mat;
//	vector<Matrix<P> > mat;
//	int row, col, channel;
//	int dim;
//
//	auto_ptr next;
//	auto_ptr prev = NULL;
//
//public:
//	placeholder(int dimention[]) {
//		dim = sizeof(dimention) / 4;
//		if (dim == 1) {
//			row = dimention[0];
//			col = 1;
//			channel = 1;
//		}
//		else if (dim == 2) {
//			row = dimention[0];
//			col = dimention[1];
//			channel = 1;
//		}
//		else if (dim == 3) {
//			row = dimention[0];
//			col = dimention[1];
//			channel = dimention[2];
//		}
//		else {
//			cout << "error in placeholder\n";
//			exit(1);
//		}
//
//		Matrix<V> temp(read, row, col, channel);
//
//		mat.push_back(temp);
//	}
//	~placeholder() {
//		for (auto it = mat.begin(); it != mat.end(); it++) {
//			it->~Matrix();
//		}
//		mat.clear();
//	}
//};
//
//template<typename V>
//class Variable {
//private:
//	float stddev;
//	int row, col;
//	int curChannels, nextChannels;
//	vector<Matrix<V> > mat;
//
//	auto_ptr prev, next;
//
//public:
//	Variable(int method, int kernel[4], float stddev) {
//		row = kernel[0];
//		col = kernel[1];
//		curChannels = kernel[2];
//		nextChannels = kernel[3];
//		this->stddev = stddev;
//
//		Matrix<V> temp(method, row, col, curChannels);
//
//		mat.push_back(temp);
//	}
//
//	~Variable() {
//		for (auto it = mat.begin(); it != mat.end(); it++) {
//			it->~Matrix();
//		}
//		mat.clear();
//	}
//};