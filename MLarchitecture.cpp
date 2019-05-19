#include <iostream>
#include <vector>
#include <memory>
#include <ctime>
#include "MLarchitecture.h"

#define max(x, y) x > y ? x : y


using namespace std;

/*
 *  �������İ��̼��� ���� �� �� ��Ʈ������ ũ�⸦ �޾ƿ;� �Ѵٸ�
 *  Layer�� weight���� �޾ƿ��� �ʰ� �� ���� Matrix���� �޾ƿ����� �Ѵ�
 */

// ������ ����
template<class M>
M RanValue() {
	M ans;

	//srand((unsigned int)time(0));

	ans = rand() / (M)RAND_MAX + 0.01;

	return ans;
}

template<class M>
Matrix<M>::Matrix(int _method, int _row, int _col, int _channels, int _type) {
	method = _method;
	row = _row;
	col = _col;
	channels = _channels;
	type = _type;

	// random�� ��� ������ �ʱ�ȭ
	if (method == random) {
		srand((unsigned int)time(0));
		mat = new M**[channels];
		for (int j = 0; j < channels; j++) {
			mat[j] = new M*[row];
			for (int k = 0; k < row; k++) {
				mat[j][k] = new M[col];
				for (int l = 0; l < col; l++)
					mat[j][k][l] = RanValue<M>();
			}
		}
	}

	// random�� �ƴ� ��� ��Ʈ���� ������ ����
	else {
		mat = new M**[channels];
		for (int j = 0; j < channels; j++) {
			mat[j] = new M*[row];
			for (int k = 0; k < row; k++) {
				mat[j][k] = new M[col];
				for (int l = 0; l < col; l++)
					mat[j][k][l] = 0;
			}
		}
	}
}

template<class M>
void Matrix<M> :: ks(int _kernel[2], int _strides[2]) {
	kernel[0] = _kernel[0];
	kernel[1] = _kernel[1];
	strides[0] = _strides[0];
	strides[1] = _strides[1];
}

template<class M>
void Matrix<M>::deleteMatrix() {
	for (int j = 0; j < channels; j++) {
		for (int k = 0; k < row; k++) {
			delete[] mat[j][k];
		}
		delete[] mat[j];
	}
	delete[] mat;
}

template<class M>
 void printmat(Matrix<M> mat) {
	 for (int ch = 0; ch < mat.channels; ch++) {
		 cout << "channel" << ch + 1 << '\n';
		 for (int i = 0; i < mat.row; i++) {
			 for (int j = 0; j < mat.col; j++) {
				 cout << mat[ch][i][j] << ' ';
			 }
			 cout << '\n';
		 }
	 }
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
			if (isrange(r + i, c + j, maxr, maxc))
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
			if (isrange(r + i, c + j, maxr, maxc) && ret < mat[r + i][c + j]) {
				
				ret = mat[r + i][c + j];
				*idx = maxr * i + j;
			}
		}
	}
	return ret;
}

Layer::Layer()
{
	row = 0;
	col = 0;
	channels = 0;
}

Layer::Layer(int method, int dimention[], int type, int len) {
	dim = len;
	strides[0] = 0;
	strides[1] = 0;
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
	Matrix<Data> temp(method, row, col, channels, type);
	matrix.push_back(temp);
}

Layer::~Layer() {
	for (auto it = this->matrix.begin(); it != this->matrix.end(); it++) {
		it->deleteMatrix();
	}
	this->matrix.clear();
}

// �ܺ���� �� �� �ִ� �Լ�
	// ��Ʈ���̵�� 0��°�� �Ʒ���, 1��°�� ����������
Matrix<Data> Layer::conv2d(Weight w, int stride[2], bool padding) {
	strides[0] = stride[0];
	strides[1] = stride[1];

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
		row = (row - w.row) / this->strides[0] + 1;
		col = (col - w.col) / this->strides[1] + 1;
		

		// ���� ��Ʈ������ ������ ����� 0���� �ʱ�ȭ
		// �̶� weight�� nextChannels�� �ܺ���� �� ����� channels
		Matrix<Data> ret(conv, row, col, w.nextChannels, conv);
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
		return ret;
	}
	// �е��� �� ��

	else {
		Matrix<Data> ret(conv, row, col, w.nextChannels, conv);
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
		return ret;
	}
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
	if ((row - kernel[0]) % strides[0]) {
		if (padding == true)
			row += strides[0] - (row - kernel[0]) % strides[0];
		else {
			cout << "error in maxpool strides\n";
			exit(1);
		}
	}
	if ((col - kernel[1]) % strides[1]) {
		if (padding == true)
			col += strides[1] - (col - kernel[1]) % strides[1];
		else {
			cout << "error in maxpool strides\n";
			exit(1);
		}
	}

	// ��Ʈ���̵忡 ���� ��� ��� ��Ʈ������ ũ��
	row = (row - kernel[0]) / strides[0] + 1;
	col = (col - kernel[1]) / strides[1] + 1;
	channels = matrix[matrix.size() - 1].channels;
	// Ǯ���ÿ� �̴� �ε����� ����
	poolingidx = Matrix<int>(maxPoolIdx, row, col, channels, maxPoolIdx);
	Matrix<Data> pool(maxPooling, row, col, channels, maxPooling);

	poolingidx.ks(kernel, strides);

	for (int i = 0; i < channels; i++) {
		int rr = 0;
		for (int r = 0; r < row; r++, rr += strides[0]) {
			int cc = 0;
			for (int c = 0; c < col; c++, cc += strides[1]) {
				pool.mat[i][r][c] = mp(matrix[matrix.size()-1].mat[i], &poolingidx.mat[i][r][c], rr, cc, matrix[matrix.size() - 1].row , matrix[matrix.size() - 1].col, kernel[0], kernel[1]);
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


Matrix<Data> Layer::Matmul(Weight w) {
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

	Matrix<Data> ret(matmul, row, w.col, w.nextChannels, matmul);

	for (int ch = 0; ch < channels; ch++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < w.col; j++) {
				int t = 0;

				for (int k = 0; k < w.col; k++) {
					t += temp.mat[ch][i][k] * w.matrix[0].mat[ch][k][i];
				}

				ret.mat[ch][i][j] = t;

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
		nextChannels = kernel[3];
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
		it->deleteMatrix();
	}
	this->matrix.clear();
}
