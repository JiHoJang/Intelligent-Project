#include <iostream>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <Windows.h>
#include "MLarchitecture.h"

using namespace std;

/*
 *  �������İ��̼��� ���� �� �� ��Ʈ������ ũ�⸦ �޾ƿ;� �Ѵٸ�
 *  Layer�� weight���� �޾ƿ��� �ʰ� �� ���� Matrix���� �޾ƿ����� �Ѵ�
 */

// ������ ����
template<class M>
M RanValue() {
	M ans;

	srand(GetTickCount64());

	ans = rand() / (M)RAND_MAX + 0.01;

	return ans;
}

template<class M>
Matrix<M>::Matrix(int _method, int _row, int _col, int _channels, int _type) {
	this->method = _method;
	this->row = _row;
	this->col = _col;
	this->channels = _channels;
	this->type = _type;

	// random�� ��� ������ �ʱ�ȭ
	if (method == random) {
		this->mat = new M**[channels]();
		for (int j = 0; j < channels; j++) {
			this->mat[j] = new M*[row]();
			for (int k = 0; k < row; k++) {
				this->mat[j][k] = new M[col]();
				for (int l = 0; l < col; l++)
					this->mat[j][k][l] = RanValue<M>();
			}
		}
	}

	// random�� �ƴ� ��� ��Ʈ���� ������ ����
	else {
		this->mat = new M**[channels]();
		for (int j = 0; j < channels; j++) {
			this->mat[j] = new M*[row]();
			for (int k = 0; k < row; k++) {
				this->mat[j][k] = new M[col]();
				for (int l = 0; l < col; l++)
					this->mat[j][k][l] = 0;
			}
		}
	}
}

template<class M>
void Matrix<M> :: ks(int _kernel[2], int _strides[2]) {
	this->kernel[0] = _kernel[0];
	this->kernel[1] = _kernel[1];
	this->strides[0] = _strides[0];
	this->strides[1] = _strides[1];
}

template<class M>
Matrix<M>::~Matrix() {
	for (int j = 0; j < channels; j++) {
		for (int k = 0; k < row; k++) {
			delete[] mat[j][k];
		}
		delete[] mat[j];
	}
	delete[] mat;
}

// �е��� ��Ʈ������ Ȯ���Ű�� �ʰ� ���� ����� ��� ������ 0���� ó��
bool isrange(int row, int col, int maxr, int maxc) {
	return row >= 0 && col >= 0 && row < maxr && col < maxc;
}


// 2���� ��� �ΰ��� �޾Ƽ� �ܺ���� �� ����
Data Conv(Data** mat1, Data** mat2, int r, int c, int maxr, int maxc, int werow, int wecol) {
	Data ret = 0;
	for (int i = 0; i < werow; i++) {
		for (int j = 0; j < wecol; j++) {
			if (isrange(r + i, c + i, maxr, maxc))
				ret += mat1[r + i][c + j] * mat2[i][j];
		}
	}
	return ret;
}

// r c �� ���� ��ġ, maxr, maxc�� mat�� �ִ� ����
// kerrow, kercol�� Ŀ���� ũ��
Data mp(Data** mat, int* idx,int r, int c, int maxr, int maxc, int kerrow, int kercol) {
	Data ret = 0;
	for (int i = 0; i < kerrow; i++) {
		for (int j = 0; j < kercol; j++) {
			if (isrange(r + i, c + i, maxr, maxc) && ret < mat[r + i][c + j]) {
				
				ret = mat[r + i][c + j];
				*idx = maxr * i + j;
			}
		}
	}
	return ret;
}

Layer::Layer(int method, int dimention[], int type, int len) {
	dim = len;
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

	// �ܺ������ ��� �϶��� ���̾ ������ ��
	// ���Ϳ��� �־��ְ� �ܺ������ ������ �Ҵ�
	if (method == conv) {
		Matrix<Data> temp;
		this->matrix.push_back(temp);
	}

	else {
		Matrix<Data> temp(method, row, col, channels, type);
	}

}

Layer::~Layer() {
	for (auto it = this->matrix.begin(); it != this->matrix.end(); it++) {
		it->~Matrix();
	}
	this->matrix.clear();
}

// �ܺ���� �� �� �ִ� �Լ�
	// ��Ʈ���̵�� 0��°�� �Ʒ���, 1��°�� ����������
Matrix<Data> Layer::conv2d(Weight w, int stride[2], bool padding) {
	strides[0] = stride[0];
	strides[1] = stride[1];

	Matrix<Data> ret;

	// �е��� ���� �ʴ� ���
	if (padding == false) {
		if ((row - w.row) % strides[0]) {
			cout << "error in conv strides\n";
			exit(1);
		}
		if ((col - w.col) % strides[1]) {
			cout << "error in conv strides\n";
			exit(1);
		}

		// ��Ʈ���̵忡 ���� ��� ��� ��Ʈ������ ũ��
		this->row = (row - w.row) / this->strides[0] + 1;
		this->col = (col - w.col) / this->strides[1] + 1;

		// ���� ��Ʈ������ ������ ����� 0���� �ʱ�ȭ
		// �̶� weight�� nextChannels�� �ܺ���� �� ����� channels
		ret = Matrix<Data> (conv, row, col, w.nextChannels, conv);
		//memset(ret.mat, 0, sizeof(ret.mat));

		// weight�� nextChannel ��ŭ ä���� ������ ��
		for (int i = 0; i < w.nextChannels; i++) {

			// ���� ä�� ������ ��� �ؾ���
			for (int j = 0; j < channels; j++) {
				// rr�� cc�� �е��� ����� �� layer matrix�� ���� ������ġ
				int rr = 0;
				//int rr = -1 * ((w.row - 1) / 2);
				for (int r = 0; r < row; r++, rr += strides[0]) {
					int cc = 0;
					//int cc = -1 * ((w.col - 1) / 2);
					for (int c = 0; c < col; c++, cc += strides[1]) {
						ret.mat[i][r][c] += Conv(matrix[matrix.size() - 1].mat[j], w.matrix[i].mat[j], rr, cc, row, col, w.row, w.col);
						// �� ���̾��� ���� ������ ��Ʈ������ �̿� (mat.size() - 1)
						// ���� �� ä�γ��� �ܺ������ ���� �׸��� ���� ä�η� �Ѿ
					}
				}
			}
		}
	}
	// �е��� �� ��

	else {
		ret = Matrix<Data> (conv, row, col, w.nextChannels, conv);
		//memset(ret.mat, 0, sizeof(ret.mat));
		for (int i = 0; i < w.nextChannels; i++) {
			for (int j = 0; j < channels; j++) {
				int rr = -1 * ((w.row - 1) / 2);
				for (int r = 0; r < row; r++, rr += strides[0]) {
					int cc = -1 * ((w.col - 1) / 2);
					for (int c = 0; c < col; c++, cc += strides[1]) {
						ret.mat[i][r][c] += Conv(matrix[matrix.size() - 1].mat[j], w.matrix[i].mat[j], rr, cc, row, col, w.row, w.col);
					}

				}
			}
		}
	}

	return ret;
}

void Layer::ReLU() {
	Matrix<Data> temp(relu, row, col, channels, relu);
	Matrix<Data> temp2 = matrix[matrix.size() - 1];
	for (int j = 0; j < channels; j++) {
		for (int r = 0; r < row; r++) {
			for (int c = 0; c < col; c++) {
				temp.mat[j][r][c] = max(0, temp2.mat[j][r][c]);
			}

		}
	}

	matrix.push_back(temp);
}

void Layer::maxPool(int kernel[2], int strides[2], bool padding) {
	// Ǯ���ÿ��� �е��� �ϴ��� 1�پ��� �ϰ� ��
	if (padding == false) {
		if ((row - kernel[0]) % strides[0]) {
			cout << "error in maxpool strides\n";
			exit(1);
		}
		if ((col - kernel[1]) % strides[1]) {
			cout << "error in maxpool strides\n";
			exit(1);
		}
	}

	// ��Ʈ���̵忡 ���� ��� ��� ��Ʈ������ ũ��
	row = (row - kernel[0]) / strides[0] + 1;
	col = (col - kernel[1]) / strides[1] + 1;

	// Ǯ���ÿ� �̴� �ε����� ����
	poolingidx = Matrix<int>(maxPoolIdx, row, col, channels, maxPoolIdx);
	Matrix<Data> pool(maxPooling, row, col, channels, maxPooling);

	poolingidx.ks(kernel, strides);

	for (int i = 0; i < channels; i++) {
		int rr = 0;
		for (int r = 0; r < row; r++, rr += strides[0]) {
			int cc = 0;
			for (int c = 0; c < col; c++, cc += strides[1]) {
				pool.mat[i][r][c] = mp(pool.mat[i], &poolingidx.mat[i][r][c], rr, cc, row, col, kernel[0], kernel[1]);
			}
		}
	}

	this->matrix.push_back(pool);
}

// ������ ���� layer�� ��Ʈ������
// ä�� 1���� row�� col���� �̷��� ��Ʈ������ ��ȯ
void Layer::Reshape(int rc[2]) {
	Matrix<Data> temp = matrix[matrix.size() - 1];

	Matrix<Data> temp2(reshape, rc[0], rc[1], 1,reshape);

	row = rc[0];
	col = rc[1];

	int r = 0;
	int c = 0;

	for (int ch = 0; ch < temp.channels; ch++) {
		for (int i = 0; i < temp.row; i++) {
			for (int j = 0; j < temp.col; j++) {
				temp2.mat[0][r][c++] = temp.mat[ch][i][j];
				if (c == rc[1]) {
					if (r == rc[0])
					{
						cout << "error in reshape";
						exit(1);
					}
					r++;
					c = 0;

				}
			}
		}
	}

	matrix.push_back(temp2);
}

template <class W>
Matrix<Data> Layer::Matmul(W w) {
	Matrix<Data> temp = matrix[matrix.size() - 1];

	if (w.nextChannels != 1) {
		cout << "Error in matmul (nextChannels)";
		exit(1);
	}
	else if (w.channels != temp.channels) {
		cout << "Error in matmul (channels)";
		exit(1);
	}
	else if (col != w.row) {
		cout << "Error in matmul (size of matrix)";
		exit(1);
	}

	Matrix<Data> ret(matmul, row, w.col, channels, matmul);

	for (int ch = 0; ch < channels; ch++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < w.col; j++) {
				int t = 0;

				for (int k = 0; k < w.col; k++) {
					t += temp[ch][i][k] * w.mat[0][ch][k][i];
				}

				ret[ch][i][j] = t;

			}
		}
	}

	return ret;
}

Weight::Weight(int method, int kernel[], int len) {
	dim = len;
	if (dim == 1) {
		row = kernel[0];
		col = 1;
		channels = 1;
		nextChannels = 1;
	}
	else if (dim == 2) {
		row = kernel[0];
		col = kernel[1];
		channels = 1;
		nextChannels = 1;
	}
	else if (dim == 3) {
		row = kernel[0];
		col = kernel[1];
		channels = kernel[2];
		nextChannels = 1;
	}
	else if (dim == 4) {
		row = kernel[0];
		col = kernel[1];
		channels = kernel[2];
		nextChannels = kernel[4];
	}
	else {
		cout << "error in placeholder\n";
		exit(1);
	}

	for (int i = 0; i < nextChannels; i++) {
		Matrix<Data> temp(method, row, col, channels, weight);
		matrix.push_back(temp);
	}
}

Weight::~Weight() {
	for (auto it = this->matrix.begin(); it != this->matrix.end(); it++) {
		it->~Matrix();
	}
	this->matrix.clear();
}
