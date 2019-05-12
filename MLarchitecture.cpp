#include <iostream>
#include <vector>
#include <memory>
#include <stdlib.h>
#include <Windows.h>
#include "MLarchitecture.h"

using namespace std;

/*
 *  백프로파게이션을 구현 할 때 매트릭스의 크기를 받아와야 한다면
 *  Layer나 weight에서 받아오지 않고 그 안의 Matrix에서 받아오도록 한다
 */

// 랜덤값 리턴
template<typename F>
F RanValue() {
	F ans;

	srand(GetTickCount());

	ans = rand() / (F)RAND_MAX + 0.01;

	return ans;
}

// layer나 weight 안에 채널, row, col을 가짐
// type은 enum 을 이용하여 현재 weight인지, conv의 결과인지, relu의 결과인지 타입을 알려줌
template<class M>
class Matrix {
private:
	M*** mat;
	int method, row, col, channels;
	int type;
	int kernel[2], strides[2];
public:
	Matrix() {}

	void ks(int _kernel[2], int _strides[2]) {
		kernel[0] = _kernel[0];
		kernel[1] = _kernel[1];
		strides[0] = _strides[0];
		strides[1] = _strides[1];
	}

	Matrix(int _method, int _row, int _col, int _channels, int _type) {
		method = _method;
		row = _row;
		col = _col;
		channels = _channels;
		type = _type;

		// random일 경우 난수로 초기화
		if (method == random) {
			mat = new(M * *)(channels);
			for (int j = 0; j < channels; j++) {
				mat[j] = new(M*)(row);
				for (int k = 0; k < row; k++) {
					mat[j][k] = new (M)(col);
					for (int l = 0; l < col; l++)
						mat[j][k][l] = RanValue<M>();
				}
			}
		}

		// random이 아닌 경우 매트릭스 구조만 생성
		else {
			mat = new(M * *)(channels);
			for (int j = 0; j < channels; j++) {
				mat[j] = new(M*)(row);
				for (int k = 0; k < row; k++) {
					mat[j][k] = new (M)(col);
					for (int l = 0; l < col; l++)
						mat[j][k][l] = 0;
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


// 패딩을 매트릭스를 확장시키지 않고 원래 행렬을 벗어난 범위는 0으로 처리
bool isrange(int row, int col, int maxr, int maxc) {
	return row >= 0 && col >= 0 && row < maxr && col < maxc;
}


// 2차원 행렬 두개를 받아서 콘볼루션 값 리턴
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

// r c 는 시작 위치, maxr, maxc는 mat의 최대 범위
// kerrow, kercol은 커널의 크기
template<class T>
T mp(T** mat, int* idx,int r, int c, int maxr, int maxc, int kerrow, int kercol) {
	T ret = 0;
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


// Layer
// Matrix의 벡터인 이유는 relu, sigmoid 등의 연산을 진행할 때 (같은 layer 안에서)
// 그 결과를 벡터의 마지막에 집어 넣어서
// 나중에 백프로파게이션 구현에 이용
template<class L>
class Layer {
private:
	vector<Matrix<L> > mat; // for calculate functions
	int row, col, channels;
	int dim;
	int strides[2];
	Matrix<int> poolingidx;

public:
	// 레이어의 앞 혹은 뒤의 weight와 연결
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
		
		// 콘볼루션의 결과 일때는 레이어를 생성한 뒤
		// 벡터에만 넣어주고 콘볼루션의 리턴을 할당
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


	// 콘볼루션 할 수 있는 함수
	// 스트라이드는 0번째가 아래로, 1번째가 오른쪽으로
	template<class We>
	Matrix<L> conv2d(We w, int strides[2], bool padding) {
		this->strides[0] = strides[0];
		this->strides[1] = strides[1];

		Matrix<L> ret;

		// 패딩을 하지 않는 경우
		if (padding == false) {
			if ((row - w.row) % strides[0]) {
				cout << "error in conv strides\n";
				exit(1);
			}
			if ((col - w.col) % strides[1]) {
				cout << "error in conv strides\n";
				exit(1);
			}

			// 스트라이드에 따른 계산 결과 매트릭스의 크기
			row = (row - w.row) / strides[0] + 1;
			col = (col - w.col) / strides[1] + 1;

			// 먼저 매트릭스의 구조를 만들고 0으로 초기화
			// 이때 weight의 nextChannels이 콘볼루션 한 결과의 channels
			ret = Matrix<L>(conv, row, col, w.nextChannels, conv);
			//memset(ret.mat, 0, sizeof(ret.mat));

			// weight의 nextChannel 만큼 채널을 만들어야 함
			for (int i = 0; i < w.nextChannels; i++) {

				// 현재 채널 끼리도 계산 해야함
				for (int j = 0; j < channels; j++) {
					// rr과 cc는 패딩을 고려할 때 layer matrix의 연산 시작위치
					int rr = 0;
					//int rr = -1 * ((w.row - 1) / 2);
					for (int r = 0; r < row; r++, rr += strides[0]) {
						int cc = 0;
						//int cc = -1 * ((w.col - 1) / 2);
						for (int c = 0; c < col; c++, cc += strides[1]) {
							ret[i][r][c] += Conv<L>(mat[mat.size()-1][j], w.mat[i][j], rr, cc, row, col, w.row, w.col);
							// 이 레이어의 가장 마지막 매트릭스를 이용 (mat.size() - 1)
							// 먼저 한 채널끼리 콘볼루션을 진행 그리고 다음 채널로 넘어감
						}
					}
				}
			}
		}
		// 패딩을 할 때
		
		else {
			ret = Matrix<L>(conv, row, col, w.nextChannels, conv);
			//memset(ret.mat, 0, sizeof(ret.mat));
			for (int i = 0; i < w.nextChannels; i++) {
				for (int j = 0; j < channels; j++) {
					int rr = -1*((w.row-1) / 2);
					for (int r = 0; r < row; r++, rr += strides[0]) {
						int cc = -1*((w.col-1) / 2);
						for (int c = 0; c < col; c++, cc += strides[1]) {
							ret[i][r][c] += Conv<L>(mat[mat.size() - 1][j], w.mat[i][j], rr, cc, row, col, w.row, w.col);
						}

					}
				}
			}
		}

		return ret;
	}

	void ReLU() {
		Matrix<L> temp(relu, row, col, channels, relu);
		Matrix<L> temp2 = mat[mat.size() - 1];
		for (int j = 0; j < channels; j++) {
			for (int r = 0; r < row; r++) {
				for (int c = 0; c < col; c++) {
					temp[j][r][c] = max(0, temp2[j][r][c]);
				}

			}
		}

		mat.push_back(temp);
	}


	
	void maxPool(int kernel[2], int strides[2], bool padding) {
		// 풀링시에는 패딩을 하더라도 1줄씩만 하게 됨
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

		// 스트라이드에 따른 계산 결과 매트릭스의 크기
		row = (row - kernel[0]) / strides[0] + 1;
		col = (col - kernel[1]) / strides[1] + 1;

		// 풀링시에 뽑는 인덱스를 저장
		poolingidx = Matrix<int>(maxPoolIdx, row, col, channels, maxPoolIdx);
		Matrix<L> pool(maxPooling, row, col, channels, maxPooling);

		poolingidx.ks(kernel, strides);

		for (int i = 0; i < channels; i++) {
			int rr = 0;
			for (int r = 0; r < row; r++, rr += strides[0]) {
				int cc = 0;
				for (int c = 0; c < col; c++, cc += strides[1] ) {
					pool[i][r][c] = mp<L>(pool[i], &poolingidx[i][r][c], rr, cc, row, col, kernel[0], kernel[1]);
				}
			}
		}

		mat.push_back(pool);
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