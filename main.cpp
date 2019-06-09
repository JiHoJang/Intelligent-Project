#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>
#include "MLarchitecture.h"

using namespace std;
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void ReadMNIST(int NumberOfImages, int DataOfAnImage, FLayer* arr, vector<Matrix<Data> >* label)
{	
	int n_rows = 0;
	int n_cols = 0;
	ifstream file("train-images.idx3-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		
		file.read((char*)& magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)& number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)& n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)& n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < NumberOfImages; ++i)
		{
			int b[3] = { 28, 28, 1 };
			Layer a(input, b, input, 3);

			//
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)& temp, sizeof(temp));
					a.matrix[0].mat[0][r][c] = (Data)temp;
				}
			}

			arr->layers.push_back(a);
		}
	}

	ifstream file2("train-labels.idx1-ubyte", ios::binary);
	if (file2.is_open()) {
		int magic_number = 0;
		int number_of_labels = 0;
		file2.read((char*)& magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

		

		file2.read((char*)& number_of_labels, sizeof(number_of_labels)), number_of_labels = ReverseInt(number_of_labels);

		//char* _dataset = new char[number_of_labels];
		for (int i = 0; i < NumberOfImages; ++i)
		{

			Matrix<Data> a(-1, 1, 10, 1, -1);



			unsigned char temp = 0;
			file2.read((char*)& temp, sizeof(temp));
			a.mat[0][0][(int)temp] = 1;


			label->push_back(a);
		}
	}
}

int main() {
	srand((unsigned int)time(0));
	FLayer Input;
	vector<Matrix<Data> > label;


	ReadMNIST(100, 784, &Input, &label);


	int w1[] = { 5, 5, 1, 32 };
	Weight W1(random, w1, 4);

	Input.next = &W1;
	W1.prev = &Input;

	FLayer L1;
	W1.next = &L1;
	L1.prev = &W1;

	int w2[] = { 5, 5, 32, 64 };
	Weight W2(random, w2, 4);
	L1.next = &W2;
	W2.prev = &L1;

	FLayer L2;
	W2.next = &L2;
	L2.prev = &W2;

	int w3[] = {7 * 7 * 64, 512 };
	Weight W3(random, w3, 2); //for matmul
	L2.next = &W3;
	W3.prev = &L2;
	
	FLayer L3;
	W3.next = &L3;
	L3.prev = &W3;
	
	int w4[] = { 512, 10 };
	Weight W4(random, w4, 2); // for matmul
	L3.next = &W4;
	W4.prev = &L3;

	FLayer L4;
	W4.next = &L4;
	L4.prev = &W4;

	int w5[] = { 1, 10 };
	Weight W5(random, w5, 2); // for add
	L4.next = &W5;
	W5.prev = &L4;

	FLayer model;
	W5.next = &model;
	model.prev = &W5;

	int batch_size = 32;
	int i[3] = { 28, 28, 1 };
	int ii[3] = { 14, 14, 1 };
	int i2[] = { 1, 512 };
	int i3[] = { 1, 10 };

	int strides1[] = { 1, 1 };
	int ker2[] = { 2, 2 };
	int rc[] = { 1, 7 * 7 * 64 };

	int b_size = 32;

	for (int t = 0; t < 100 / 32; t++) {
		for (int batch = 0; batch < 32; batch++) {
			Layer temp(conv, i, conv, 3);
			temp.matrix.push_back(Input.layers[t + batch].conv2d(W1, strides1, true));
			temp.ReLU();
			temp.maxPool(ker2, ker2, true);
			L1.layers.push_back(temp);

			Layer temp2(conv, ii, conv, 3);
			temp2.matrix.push_back(L1.layers[L1.layers.size() - 1].conv2d(W2, strides1, true));
			temp2.ReLU();
			temp2.maxPool(ker2, ker2, true);
			temp2.Reshape(rc);
			L2.layers.push_back(temp2);

			Layer temp3(matmul, i2, matmul, 2);
			temp3.matrix.push_back(L2.layers[L2.layers.size() - 1].Matmul(W3));
			temp3.ReLU();
			L3.layers.push_back(temp3);

			Layer temp4(matmul, i3, matmul, 2);
			temp4.matrix.push_back(L3.layers[L3.layers.size() - 1].Matmul(W4));
			L4.layers.push_back(temp4);

			Layer temp5(add, i3, add, 2);
			temp5.matrix.push_back(L4.layers[L4.layers.size() - 1].Add(W5));
			temp5.SoftMax();
			model.layers.push_back(temp5);
		}
		FLayer* pointer = &model;
		while (true) {
			pointer->backPropagation(0.01, label);
			pointer->index += b_size;
			pointer->batch_size = b_size;

			if (pointer->prev == NULL)
				break;
			pointer = pointer->prev->prev;
		}
		
	}

	system("pause");
}
