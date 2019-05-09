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
template<typename F>
F RanValue(int method) {
	F ans;

	srand(GetTickCount());

	ans = rand() / (F)RAND_MAX + 0.01;

	return ans;
}

// layer�� weight �ȿ� ä��, row, col�� ����
// type�� enum �� �̿��Ͽ� ���� weight����, conv�� �������, relu�� ������� Ÿ���� �˷���
template<class M>
class Matrix {
private:
	M*** mat;
	int method, row, col, channels;
	int type;

public:
	Matrix() {}

	Matrix(int _method, int _row, int _col, int _channels, int _type) {
		method = _method;
		row = _row;
		col = _col;
		channels = _channels;
		type = _type;

		// random�� ��� ������ �ʱ�ȭ
		if (method == random) {
			mat = new(M * *)(channels);
			for (int j = 0; j < channels; j++) {
				mat[j] = new(M*)(row);
				for (int k = 0; k < row; k++) {
					mat[j][k] = new (M)(col);
					for (int l = 0; l < col; l++)
						mat[j][k][l] = RanValue<M>(method);
				}
			}
		}

		// random�� �ƴ� ��� ��Ʈ���� ������ ����
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


// �е��� ��Ʈ������ Ȯ���Ű�� �ʰ� ���� ����� ��� ������ 0���� ó��
bool isrange(int row, int col, int maxr, int maxc) {
	return row >= 0 && col >= 0 && row < maxr && col < maxc;
}


// 2���� ��� �ΰ��� �޾Ƽ� �ܺ���� �� ����
template<class T>
T Conv(T** mat1, T** mat2, int r, int c, int maxr, int maxc, int werow, int wecol) {
	T ret = 0;
	for (int i = 0; i < werow; i++) {
		for (int j = 0; j < wecol; j++) {
			if (isrange(r + i, c + i, maxr, maxc))
				ret += mat1[r + i][c + j] * mat2[i][j];
		}
	}
	return ret;
}


// Layer
// Matrix�� ������ ������ relu, sigmoid ���� ������ ������ �� (���� layer �ȿ���)
// �� ����� ������ �������� ���� �־
// ���߿� �������İ��̼� ������ �̿�
template<class L>
class Layer {
private:
	vector<Matrix<L> > mat; // for calculate functions
	int row, col, channels;
	int dim;
	int strides[2];

public:
	// ���̾��� �� Ȥ�� ���� weight�� ����
	Weight<L>* prev = NULL;
	Weight<L>* next = NULL;

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
		
		// �ܺ������ ��� �϶��� ���̾ ������ ��
		// ���Ϳ��� �־��ְ� �ܺ������ ������ �Ҵ�
		if (method == conv) {
			Matrix<L> temp();
			mat.push_back(temp);
		}

		else {
			Matrix<L> temp(method, row, col, channels, type);
		}
		
	}

	~Layer() {
		for (auto it = mat.begin(); it != mat.end(); it++) {
			it->~Matrix();
		}
		mat.clear();
	}


	// �ܺ���� �� �� �ִ� �Լ�
	// ��Ʈ���̵�� 0��°�� �Ʒ���, 1��°�� ����������
	template<class We>
	Matrix<L> conv2d(We w, int strides[2], bool padding) {
		this->strides[0] = strides[0];
		this->strides[1] = strides[1];

		Matrix<L> ret;

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
			row = (row - w.row) / strides[0] + 1;
			col = (col - w.col) / strides[1] + 1;

			// ���� ��Ʈ������ ������ ����� 0���� �ʱ�ȭ
			// �̶� weight�� nextChannels�� �ܺ���� �� ����� channels
			ret = Matrix<L>(conv, row, col, w.nextChannels, conv);
			memset(ret, 0, sizeof(ret));

			// weight�� nextChannel ��ŭ ä���� ������ ��
			for (int i = 0; i < w.nextChannels; i++) {

				// ���� ä�� ������ ��� �ؾ���
				for (int j = 0; j < channels; j++) {
					// rr�� cc�� �е��� ����� �� layer matrix�� ���� ������ġ
					int rr = 0;
					for (int r = 0; r < row; r++, rr += strides[0]) {
						int cc = 0;
						for (int c = 0; c < col; c++, cc += strides[1]) {
							ret[i][r][c] += Conv<L>(mat[mat.size()-1][j], w.mat[i][j], rr, cc, row, col, w.row, w.col);
							// �� ���̾��� ���� ������ ��Ʈ������ �̿� (mat.size() - 1)
							// ���� �� ä�γ��� �ܺ������ ���� �׸��� ���� ä�η� �Ѿ
						}

					}
				}
			}
		}
		// �̴� �е��� �� ��
		else {
			ret = Matrix<L>(conv, row, col, w.nextChannels, conv);
			memset(ret, 0, sizeof(ret));
			for (int i = 0; i < w.nextChannels; i++) {
				for (int j = 0; j < channels; j++) {
					int rr = (row-1) / 2;
					for (int r = 0; r < row; r++, rr += strides[0]) {
						int cc = (col-1) / 2;
						for (int c = 0; c < col; c++, cc += strides[1]) {
							ret[i][r][c] += Conv<L>(mat[mat.size() - 1][j], w.mat[i][j], rr, cc, row, col, w.row, w.col);
						}

					}
				}
			}
		}

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
	Layer<W>* prev = NULL;
	Layer<W>* next = NULL;

	Weight(int method, int kernel[4]) {
		row = kernel[0];
		col = kernel[1];
		channels = kernel[2];
		nextChannels = kernel[3];

		for (int i = 0; i < nextChannels; i++) {
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
};