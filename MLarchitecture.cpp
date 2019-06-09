#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "MLarchitecture.h"

#define max(x, y) x > y ? x : y


using namespace std;

/*
 *  백프로파게이션을 구현 할 때 매트릭스의 크기를 받아와야 한다면
 *  Layer나 weight에서 받아오지 않고 그 안의 Matrix에서 받아오도록 한다
 */

 // 랜덤값 리턴
template<class M>
M RanValue() {
	M ans;

	//srand((unsigned int)time(0));

	ans = rand() / (M)RAND_MAX - 0.5;


	return ans*0.5;
}

template<class M>
Matrix<M>::Matrix(int _method, int _row, int _col, int _channels, int _type) {
	method = _method;
	row = _row;
	col = _col;
	channels = _channels;
	type = _type;
	kernel[0] = 0;
	kernel[1] = 0;
	strides[0] = 0;
	strides[1] = 0;

	// random일 경우 난수로 초기화
	if (method == random) {
		mat = new M * *[channels];
		for (int j = 0; j < channels; j++) {
			mat[j] = new M * [row];
			for (int k = 0; k < row; k++) {
				mat[j][k] = new M[col];
				for (int l = 0; l < col; l++)
					mat[j][k][l] = RanValue<M>();
			}
		}
	}
	else if (method == update) {
		mat = new M * *[channels];
		for (int j = 0; j < channels; j++) {
			mat[j] = new M * [row];
			for (int k = 0; k < row; k++) {
				mat[j][k] = new M[col];
				for (int l = 0; l < col; l++)
					mat[j][k][l] = 1;
			}
		}
	}
	// random이 아닌 경우 매트릭스 구조만 생성
	else {
		mat = new M * *[channels];
		for (int j = 0; j < channels; j++) {
			mat[j] = new M * [row];
			for (int k = 0; k < row; k++) {
				mat[j][k] = new M[col];
				for (int l = 0; l < col; l++)
					mat[j][k][l] = 0;
			}
		}
	}
}

template<class M>
void Matrix<M> ::ks(int _kernel[2], int _strides[2]) {
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

// 패딩을 매트릭스를 확장시키지 않고 원래 행렬을 벗어난 범위는 0으로 처리
bool isrange(int row, int col, int maxr, int maxc) {
	return row >= 0 && col >= 0 && row < maxr && col < maxc;
}


// 2차원 행렬 두개를 받아서 콘볼루션 값 리턴
Data Conv(Data * *mat1, Data * *mat2, int r, int c, int maxr, int maxc, int werow, int wecol) {
	Data ret = 0;
	for (int i = 0; i < werow; i++) {
		for (int j = 0; j < wecol; j++) {
			if (isrange(r + i, c + j, maxr, maxc))
				ret += mat1[r + i][c + j] * mat2[i][j];
		}
	}
	return ret;
}

// r c 는 시작 위치, maxr, maxc는 mat의 최대 범위
// kerrow, kercol은 커널의 크기
Data mp(Data * *mat, int* idx, int r, int c, int maxr, int maxc, int kerrow, int kercol) {
	Data ret = 0;
	for (int i = 0; i < kerrow; i++) {
		for (int j = 0; j < kercol; j++) {
			if (isrange(r + i, c + j, maxr, maxc) && ret < mat[r + i][c + j]) {
				ret = mat[r + i][c + j];
				*idx = maxc * (r + i) + c + j;
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

	// 콘볼루션의 결과 일때는 레이어를 생성한 뒤
	// 벡터에만 넣어주고 콘볼루션의 리턴을 할당
	if (method == conv || method == matmul || method == add) {
		//Matrix<Data> temp;
		//matrix.push_back(temp);
	}
	else {
		Matrix<Data> temp(method, row, col, channels, type);
		matrix.push_back(temp);
	}
}


// 콘볼루션 할 수 있는 함수
	// 스트라이드는 0번째가 아래로, 1번째가 오른쪽으로
Matrix<Data> Layer::conv2d(Weight w, int stride[2], bool padding) {
	strides[0] = stride[0];
	strides[1] = stride[1];

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
		row = (row - w.row) / this->strides[0] + 1;
		col = (col - w.col) / this->strides[1] + 1;


		// 먼저 매트릭스의 구조를 만들고 0으로 초기화
		// 이때 weight의 nextChannels이 콘볼루션 한 결과의 channels
		Matrix<Data> ret(conv, row, col, w.nextChannels, conv);
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
						ret.mat[i][r][c] += Conv(matrix[matrix.size() - 1].mat[j], w.matrix[i].mat[j], rr, cc, row, col, w.row, w.col);
						// 이 레이어의 가장 마지막 매트릭스를 이용 (mat.size() - 1)
						// 먼저 한 채널끼리 콘볼루션을 진행 그리고 다음 채널로 넘어감
					}
				}
			}
		}
		return ret;
	}
	// 패딩을 할 때

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
	// 풀링시에는 패딩을 하더라도 1줄씩만 하게 됨
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

	// 스트라이드에 따른 계산 결과 매트릭스의 크기
	row = (row - kernel[0]) / strides[0] + 1;
	col = (col - kernel[1]) / strides[1] + 1;
	channels = matrix[matrix.size() - 1].channels;
	// 풀링시에 뽑는 인덱스를 저장
	poolingidx = Matrix<int>(maxPoolIdx, row, col, channels, maxPoolIdx);
	Matrix<Data> pool(maxPooling, row, col, channels, maxPooling);

	poolingidx.ks(kernel, strides);

	for (int i = 0; i < channels; i++) {
		int rr = 0;
		for (int r = 0; r < row; r++, rr += strides[0]) {
			int cc = 0;
			for (int c = 0; c < col; c++, cc += strides[1]) {
				pool.mat[i][r][c] = mp(matrix[matrix.size() - 1].mat[i], &poolingidx.mat[i][r][c], rr, cc, matrix[matrix.size() - 1].row, matrix[matrix.size() - 1].col, kernel[0], kernel[1]);
			}
		}
	}

	this->matrix.push_back(pool);
}

// 마지막 계산된 layer의 매트릭스를
// 채널 1개의 row와 col으로 이뤄진 매트릭스로 변환
void Layer::Reshape(int rc[2]) {
	Matrix<Data> temp = matrix[matrix.size() - 1];

	Matrix<Data> temp2(reshape, rc[0], rc[1], 1, reshape);

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
	else if (temp.col != w.row) {
		cout << "Error in matmul (size of matrix)";
		exit(1);
	}

	Matrix<Data> ret(matmul, temp.row, w.col, w.nextChannels, matmul);

	for (int ch = 0; ch < temp.channels; ch++) {
		for (int i = 0; i < temp.row; i++) {
			for (int j = 0; j < w.col; j++) {
				Data t = 0;

				for (int k = 0; k < w.col; k++) {
					t += temp.mat[ch][i][k] * w.matrix[0].mat[ch][k][j];
				}

				ret.mat[ch][i][j] = t;

			}
		}
	}

	return ret;
}


Matrix<Data> Layer::Add(Weight w) {
	Matrix<Data> temp = matrix[matrix.size() - 1];

	if (w.nextChannels != 1) {
		cout << "Error in matmul (nextChannels)";
		exit(1);
	}
	else if (w.channels != temp.channels) {
		cout << "Error in matmul (channels)";
		exit(1);
	}
	else if (temp.row != w.row) {
		cout << "Error in matmul (size of matrix)";
		exit(1);
	}
	else if (temp.col != w.col) {
		cout << "Error in matmul (size of matrix)";
		exit(1);
	}

	Matrix<Data> ret(add, temp.row, w.col, w.nextChannels, add);

	for (int ch = 0; ch < temp.channels; ch++) {
		for (int i = 0; i < temp.row; i++) {
			for (int j = 0; j < temp.col; j++) {

				ret.mat[ch][i][j] = temp.mat[ch][i][j] + w.matrix[0].mat[ch][i][j];

			}
		}
	}

	return ret;
}

void Layer::SoftMax() {
	row = matrix[matrix.size() - 1].row;
	col = matrix[matrix.size() - 1].col;
	channels = matrix[matrix.size() - 1].channels;

	int num = row * col * channels;
	double sum = 0.0;

	for (int ch = 0; ch < channels; ch++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				sum += (Data)exp(matrix[matrix.size() - 1].mat[ch][i][j]);
			}
		}
	}

	Matrix<Data> temp(softmax, row, col, channels, softmax);

	for (int ch = 0; ch < channels; ch++) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				temp.mat[ch][i][j] = (Data)(exp(matrix[matrix.size() - 1].mat[ch][i][j]) / sum);
			}
		}
	}

	matrix.push_back(temp);
}

Data MinMax(Data num, Data low, Data high) {
	if (num > high) return high;
	else if (num < low) return low;
	return num;
}



void Layer::LError(Matrix<Data> label) {
	int row = matrix[matrix.size() - 1].row;
	int col = matrix[matrix.size() - 1].col;
	int channels = matrix[matrix.size() - 1].channels;

	if (label.row != row || label.col != col || label.channels != channels) {
		printf("error in cost \n");
		exit(1);
	}

	Data answer = 0;

	Matrix<Data> ans(error, 1, 1, 1, error);
	Matrix<Data> temp = matrix[matrix.size() - 1];

	for (int ch = 0; ch < channels; ch++)
		for (int r = 0; r < row; r++)
			for (int c = 0; c < col; c++)
				answer += label.mat[ch][r][c] * log(MinMax(temp.mat[ch][r][c], 0.0000000001, 1));
	ans.mat[0][0][0] = answer;
	matrix.push_back(ans);
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
		Matrix<Data> temp2(-1, row, col, channels, -1);
		updateTemp.push_back(temp2);
	}
}
//
//Weight::~Weight() {
//	auto it2 = this->updateTemp.begin();
//	for (auto it = this->matrix.begin(); it != this->matrix.end(); it++) {
//		it->deleteMatrix();
//		it2->deleteMatrix();
//		it2++;
//	}
//	this->matrix.clear();
//	this->updateTemp.clear();
//}

FLayer::FLayer() {
	index = 0;
}

void FLayer::backPropagation(vector<Matrix<Data> > label) {
	if (prev == NULL) return;

	// 레이어 안에서 연산된 크기
	int size = layers[0].matrix.size();
	// 배치 사이즈 만큼 각 레이어에 대해
	for (int i = index; i < index + batch_size; i++) {
		for (int j = size - 1; j >= 0; j--) {
			if (layers[i].backprop.size() == 0) {
				Matrix<Data> temp(update, layers[i].matrix[j].row, layers[i].matrix[j].col, layers[i].matrix[j].channels, update);
				if (next != NULL) {
					if (next->next->layers[i].matrix[next->next->layers[i].matrix.size() - 1].type == add) {
						for (int ch = 0; ch < temp.channels; ch++) {
							for (int r = 0; r < temp.row; r++) {
								for (int c = 0; c < temp.col; c++) {
									temp.mat[ch][r][c] = next->next->layers[i].backprop[next->next->layers[i].backprop.size() - 1].mat[ch][r][c];
								}
							}
						}
					}
					else {
						for (int ch = 0; ch < temp.channels; ch++) {
							for (int r = 0; r < temp.row; r++) {
								for (int c = 0; c < temp.col; c++) {
									temp.mat[ch][r][c] = next->next->layers[i].backprop[next->next->layers[i].backprop.size() - 1].mat[ch][r][c] * next->matrix[0].mat[ch][r][c];
								}
							}
						}
					}

				}
				layers[i].backprop.push_back(temp);
			}
			int flag = layers[i].matrix[j].type;
			// add
			if (flag == add) {
				for (int l = 0; l < prev->channels; l++) {
					for (int r = 0; r < prev->row; r++) {
						for (int c = 0; c < prev->col; c++) {
							prev->updateTemp[0].mat[l][r][c] += layers[i].backprop[layers[i].backprop.size() - 1].mat[l][r][c];
							//prev->updateTemp[k].mat[l][r][c] = prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].mat[0][(k*l*r*c + l*r*c + r*c + c) / prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].col][(k * l * r * c + l * r * c + r * c + c) % prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].col] * layers[i].backprop[layers[i].backprop.size()-1].mat[l][r][c];
						}
					}
				}
			}
			else if (flag == softmax) {
				for (int l = 0; l < layers[i].matrix[j].channels; l++) {
					for (int r = 0; r < layers[i].matrix[j].row; r++) {
						for (int c = 0; c < layers[i].matrix[j].col; c++) {
							layers[i].backprop[layers[i].backprop.size() - 1].mat[l][r][c] *= layers[i].matrix[j].mat[l][r][c] - label[i].mat[l][r][c];
						}
					}
				}

			}
			// matmul 다시 하기
			else if (flag == matmul) {
				for (int l = 0; l < prev->channels; l++) {
					for (int r = 0; r < prev->row; r++) {
						for (int c = 0; c < prev->col; c++) {
							Data temp = 0;
							for (int k = 0; k < prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].row; k++) {
								temp += prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].mat[l][k][r] * layers[i].backprop[layers[i].backprop.size() - 1].mat[l][k][c];
							}
							prev->updateTemp[0].mat[l][r][c] += temp;
						}
					}
				}
				int channel = prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].channels;
				int row = prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].row;
				int col = prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].col;

				Matrix<Data> temp(update, row, col, channel, update);
				for (int ch = 0; ch < channel; ch++) {
					for (int r = 0; r < row; r++) {
						for (int c = 0; c < col; c++) {
							Data temp2 = 0;

							for (int k = 0; k < prev->matrix[0].col; k++) {
								temp2 += prev->matrix[0].mat[ch][c][k] * layers[i].backprop[layers[i].backprop.size() - 1].mat[ch][r][k];
							}
							temp.mat[ch][r][c] = temp2;
						}
					}
				}
				prev->prev->layers[i].backprop.push_back(temp);
			}
			else if (flag == reshape) {
				// 다시 이전꺼로 바꿔주고
				int channel = layers[i].matrix[j - 1].channels;
				int row = layers[i].matrix[j - 1].row;
				int col = layers[i].matrix[j - 1].col;

				Matrix<Data> temp(update, row, col, channel, update);

				int l = 0;
				int m = 0;
				int n = 0;

				for (int ch = 0; ch < channel; ch++) {
					for (int r = 0; r < row; r++) {
						for (int c = 0; c < col; c++) {
							temp.mat[ch][r][c] = layers[i].backprop[layers[i].backprop.size() - 1].mat[l][m][n++];
							if (n == layers[i].backprop[layers[i].backprop.size() - 1].col) {
								n = 0;
								m++;
								if (m == layers[i].backprop[layers[i].backprop.size() - 1].row) {
									m = 0;
									l++;
								}
							}
						}
					}
				}

				layers[i].backprop.push_back(temp);
			}
			else if (flag == maxPooling) {
				int channel = layers[i].matrix[j - 1].channels;
				int row = layers[i].matrix[j - 1].row;
				int col = layers[i].matrix[j - 1].col;
				Matrix<Data> temp(-1, row, col, channel, -1);

				Matrix<int> idx = layers[i].poolingidx;

				for (int ch = 0; ch < channel; ch++) {
					for (int r = 0; r < layers[i].matrix[j].row; r++) {
						for (int c = 0; c < layers[i].matrix[j].col; c++) {
							temp.mat[ch][idx.mat[ch][r][c] / col][idx.mat[ch][r][c] % col] = layers[i].backprop[layers[i].backprop.size() - 1].mat[ch][r][c];
						}
					}
				}

				layers[i].backprop.push_back(temp);
			}

			else if (flag == conv) {
				//ㅅㅂ...
				prev->updateTemp.clear();
				int midr = prev->row / 2;
				int midc = prev->col / 2;

				for (int nch = 0; nch < prev->nextChannels; nch++) {
					Matrix<Data> temp(-1, prev->row, prev->col, prev->channels, -1);
					for (int ch = 0; ch < prev->channels; ch++) {
						for (int r = 0; r < prev->row; r++) {
							for (int c = 0; c < prev->col; c++) {
								Data temp2 = 0;

								for (int y = 0; y < prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].row; y++) {
									for (int z = 0; z < prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].col; z++) {
										if (isrange(y + midr - r, z + midc - c, prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].row, prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].col)) {
											temp2 += layers[i].backprop[layers[i].backprop.size() - 1].mat[nch][y + midr - r][z + midc - c] * prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].mat[ch][y][z];
										}
									}
								}

								temp.mat[ch][r][c] = temp2;
							}
						}
					}
					prev->updateTemp.push_back(temp);
				}

				Matrix<Data> temp(-1, prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].row, prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].col, prev->prev->layers[i].matrix[prev->prev->layers[i].matrix.size() - 1].channels, -1);

				for (int ch = 0; ch < temp.channels; ch++) {
					for (int r = 0; r < temp.row; r++) {
						for (int c = 0; c < temp.col; c++) {
							Data temp2 = 0;

							for (int nch = 0; nch < prev->nextChannels; nch++) {
								for (int x = 0; x < prev->row; x++) {
									for (int y = 0; y < prev->col; y++) {
										if (isrange(r + midr - x, c + midc - y, temp.row, temp.col))
											temp2 += prev->matrix[nch].mat[ch][x][y] * layers[i].backprop[layers[i].backprop.size() - 1].mat[nch][r + midr - x][c + midc - y];
									}
								}
							}

							temp.mat[ch][r][c];
						}
					}
				}

				prev->prev->layers[i].backprop.push_back(temp);

			}
			else if (flag == sigmoid) {
				for (int l = 0; l < layers[i].matrix[j].channels; l++) {
					for (int r = 0; r < layers[i].matrix[j].row; r++) {
						for (int c = 0; c < layers[i].matrix[j].col; c++) {
							layers[i].backprop[layers[i].backprop.size() - 1].mat[l][r][c] *= layers[i].matrix[j].mat[l][r][c] * (1 - layers[i].matrix[j].mat[l][r][c]);
						}
					}
				}
			}
			else if (flag == relu) {
				for (int l = 0; l < layers[i].matrix[j].channels; l++) {
					for (int r = 0; r < layers[i].matrix[j].row; r++) {
						for (int c = 0; c < layers[i].matrix[j].col; c++) {
							if (layers[i].matrix[j].mat[l][r][c] <= 0)
								layers[i].backprop[layers[i].backprop.size() - 1].mat[l][r][c] = 0;
						}
					}
				}
			}
		}
	}
	//train(learningRate);
}

void FLayer::train(float learningRate) {
	Weight* pointer = prev;
	while (pointer != NULL) {
		for (int nch = 0; nch < pointer->nextChannels; nch++) {
			for (int ch = 0; ch < pointer->channels; ch++) {
				for (int r = 0; r < pointer->row; r++) {
					for (int c = 0; c < pointer->col; c++) {
						pointer->matrix[nch].mat[ch][r][c] -= pointer->updateTemp[nch].mat[ch][r][c] * learningRate;
					}
				}
			}
		}

		pointer->updateTemp.clear();
		Matrix<Data> temp(-1, pointer->row, pointer->col, pointer->channels, -1);
		pointer->updateTemp.push_back(temp);

		if (pointer->prev == NULL) break;
		pointer = pointer->prev->prev;
	}
}

int argMax(Matrix<Data> mat) {
	Data m = 0;
	int idx = -1;
	for (int i = 0; i < mat.col; i++) {
		if (mat.mat[0][0][i] > m) {
			m = mat.mat[0][0][i];
			idx = i;
		}
	}

	if (idx == -1) {
		cout << "Error in argmax\n";
		exit(1);
	}
	return idx;
}

float FLayer::accuracy(vector<Matrix<Data> > label) {
	float ans = 0;
	for (int i = index; i < index + batch_size; i++) {
		if (argMax(layers[i].matrix[layers[i].matrix.size() - 1]) == argMax(label[i])) {
			ans += 1;
		}
		//cout << argMax(layers[i].matrix[layers[i].matrix.size() - 1]) << ' ' << argMax(label[i]) << '\n';
	}
	return ans / (float)batch_size;
}