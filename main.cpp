#include <iostream>
#include <ctime>
#include "MLarchitecture.h"

using namespace std;

int main() {
	srand((unsigned int)time(0));
	int inp[3] = { 5, 5, 3 };
	int np[] = { 3, 3, 3, 5 };
	int stride[2] = { 1, 1 };

	Weight W(random, np, 4);
	Layer X(random, inp, input, 3);

	for (int ch = 0; ch < X.matrix[0].channels; ch++) {
		cout << "channel" << ch + 1 << '\n';
		for (int i = 0; i < X.matrix[0].row; i++) {
			for (int j = 0; j < X.matrix[0].col; j++) {
				cout << X.matrix[0].mat[ch][i][j] << ' ';
			}
			cout << '\n';
		}
	}

	cout << "\nWeight\n";
	for (int nch = 0; nch < W.nextChannels; nch++) {
		cout << "nextChannel" << nch + 1 << '\n';
		for (int ch = 0; ch < W.matrix[nch].channels; ch++) {
			cout << "channel" << ch + 1 << '\n';
			for (int i = 0; i < W.matrix[nch].row; i++) {
				for (int j = 0; j < W.matrix[nch].col; j++) {
					cout << W.matrix[nch].mat[ch][i][j] << ' ';
				}
				cout << '\n';
			}
		}
	}

	cout << "\nConv\n";
	Layer X2(conv, inp, conv, 3);
	X2.matrix.push_back(X.conv2d(W, stride, true));

	for (int ch = 0; ch < X2.matrix[0].channels; ch++) {
		cout << "channel" << ch + 1 << '\n';
		for (int i = 0; i < X2.matrix[0].row; i++) {
			for (int j = 0; j < X2.matrix[0].col; j++) {
				cout << X2.matrix[0].mat[ch][i][j] << ' ';
			}
			cout << '\n';
		}
	}

	cout << "\n Pooling \n";

	int kernel[] = { 3, 3 };
	stride[0] = 3;
	stride[1] = 3;

	X2.maxPool(kernel, stride, true);

	for (int ch = 0; ch < X2.matrix[1].channels; ch++) {
		cout << "channel" << ch + 1 << '\n';
		for (int i = 0; i < X2.matrix[1].row; i++) {
			for (int j = 0; j < X2.matrix[1].col; j++) {
				cout << X2.matrix[1].mat[ch][i][j] << ' ';
			}
			cout << '\n';
		}
	}

	int rc[2] = { 1, 2 * 2 * 5 };
	X2.Reshape(rc);

	for (int ch = 0; ch < X2.matrix[2].channels; ch++) {
		cout << "channel" << ch + 1 << '\n';
		for (int i = 0; i < X2.matrix[2].row; i++) {
			for (int j = 0; j < X2.matrix[2].col; j++) {
				cout << X2.matrix[2].mat[ch][i][j] << ' ';
			}
			cout << '\n';
		}
	}

	int w2[2] = { 20, 10 };
	Weight W2(random, w2, 2);


	int m[2] = { 1, 10 };
	Layer model(matmul, m, matmul, 2);
	model.matrix.push_back(X2.Matmul(W2));

	cout << "\n Matmul \n";

	for (int ch = 0; ch < model.matrix[0].channels; ch++) {
		cout << "channel" << ch + 1 << '\n';
		for (int i = 0; i < model.matrix[0].row; i++) {
			for (int j = 0; j < model.matrix[0].col; j++) {
				cout << model.matrix[0].mat[ch][i][j] << ' ';
			}
			cout << '\n';
		}
	}


	cout << "\n Softmax \n";

	model.SoftMax();

	for (int ch = 0; ch < model.matrix[1].channels; ch++) {
		cout << "channel" << ch + 1 << '\n';
		for (int i = 0; i < model.matrix[1].row; i++) {
			for (int j = 0; j < model.matrix[1].col; j++) {
				cout << model.matrix[1].mat[ch][i][j] << ' ';
			}
			cout << '\n';
		}
	}
	system("pause");
}
