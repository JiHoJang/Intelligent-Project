#ifndef MLarchitecture_h
#define MLarchitecture_h
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
	maxPooling,
	maxPoolIdx,
	reshape,
	matmul,
};

typedef float Data;

class Weight;

// layer나 weight 안에 채널, row, col을 가짐
// type은 enum 을 이용하여 현재 weight인지, conv의 결과인지, relu의 결과인지 타입을 알려줌
template<class M>
class Matrix {
public:
	int method, row, col, channels;
	int type;
	int kernel[2];
	int	strides[2];

	M*** mat;
	Matrix() {
		row = 0;
		col = 0;
		channels = 0;
	}


	Matrix(int _method, int _row, int _col, int _channels, int _type);

	void ks(int _kernel[2], int _strides[2]);

	void deleteMatrix();
	//~Matrix();
};

template<class M>
void printmat(Matrix<M> mat);

// Layer
// Matrix의 벡터인 이유는 relu, sigmoid 등의 연산을 진행할 때 (같은 layer 안에서)
// 그 결과를 벡터의 마지막에 집어 넣어서
// 나중에 백프로파게이션 구현에 이용
class Layer {
public:
	vector<Matrix<Data> > matrix; // for calculate functions
	int row, col, channels;
	int dim;
	int strides[2] = { 0 };
	Matrix<int> poolingidx;

	// 레이어의 앞 혹은 뒤의 weight와 연결
	Weight* prev = NULL;
	Weight* next = NULL;

	Layer();

	Layer(int method, int dimention[], int type, int len);

	~Layer();

	Matrix<Data> conv2d(Weight w, int stride[2], bool padding);

	void ReLU();

	void maxPool(int kernel[2], int strides[2], bool padding);

	void Reshape(int rc[2]);


	// 매트릭스 간의 곱
	Matrix<Data> Matmul(Weight w);

};

class Weight {
public:
	vector< Matrix<Data> > matrix; // for next channels
	int row, col, channels;
	int nextChannels;
	int dim;

	Layer* prev = NULL;
	Layer* next = NULL;

	Weight() {
		row = 0;
		col = 0;
		channels = 0;
		nextChannels = 0;
	}

	Weight(int method, int kernel[], int len);

	~Weight();
};

#endif