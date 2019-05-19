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

// layer�� weight �ȿ� ä��, row, col�� ����
// type�� enum �� �̿��Ͽ� ���� weight����, conv�� �������, relu�� ������� Ÿ���� �˷���
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
// Matrix�� ������ ������ relu, sigmoid ���� ������ ������ �� (���� layer �ȿ���)
// �� ����� ������ �������� ���� �־
// ���߿� �������İ��̼� ������ �̿�
class Layer {
public:
	vector<Matrix<Data> > matrix; // for calculate functions
	int row, col, channels;
	int dim;
	int strides[2] = { 0 };
	Matrix<int> poolingidx;

	// ���̾��� �� Ȥ�� ���� weight�� ����
	Weight* prev = NULL;
	Weight* next = NULL;

	Layer();

	Layer(int method, int dimention[], int type, int len);

	~Layer();

	Matrix<Data> conv2d(Weight w, int stride[2], bool padding);

	void ReLU();

	void maxPool(int kernel[2], int strides[2], bool padding);

	void Reshape(int rc[2]);


	// ��Ʈ���� ���� ��
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