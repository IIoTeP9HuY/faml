#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <iomanip>
#include <tuple>
#include <memory>

#include "faml/data/table.hpp"
#include "faml/kernels.hpp"
#include "faml/distances.hpp"
#include "faml/io.hpp"
#include "faml/models/knn.hpp"
#include "faml/preprocessing/scaler.hpp"
#include "faml/cross_validation/cross_validation.hpp"
#include "faml/quality/classification.hpp"

using namespace std;
using namespace faml;
using namespace Eigen;

int main() {
	auto testData = readCSV("mnist_small_train.csv");
	auto testStr = readCSV("mnist_test_no_label.csv");
	typedef vector<string> StrRowType;
	typedef unsigned long long Label;
	Table<StrRowType> trainXstr, trainYstr;
	tie(trainXstr, trainYstr) = testData.splitOnColumns({"label"});
	auto trainY = trainYstr.cast(
		[](const StrRowType &sample) { 
			return stoull(sample[0]);
		}
	);

	auto trainX = trainXstr.castByElement<VectorXf>(
		[](const string& x) {
			return stod(x);
		}
	);
	cerr << "before" << endl;
	int size = 28;
	trainX = trainX.cast(
		[](const VectorXf& v) {
			VectorXf res = VectorXf::Zero(14 * 14);
			for(int i = 0; i < 14; ++i) {
				for(int j = 0; j < 14; ++j) {
					for(int ii = 0; ii < 2; ++ii) {
						for(int jj = 0; jj < 2; ++jj) {
							res[i * 14 + j] += v[(2 * i + ii) * 28 + 2 * j + jj];
						}
					}
				}
			}
			return res;
		}
	);
	 
	size /= 2;

	cerr << "after" << endl;
	auto test = testStr.castByElement<VectorXf>(
		[](const std::string& x) {
			return std::stod(x);
		}
	);

	
	test = test.cast(
		[](const VectorXf& v) {
			VectorXf res = VectorXf::Zero(14 * 14);
			for(int i = 0; i < 14; ++i) {
				for(int j = 0; j < 14; ++j) {
					for(int ii = 0; ii < 2; ++ii) {
						for(int jj = 0; jj < 2; ++jj) {
							res[i * 14 + j] += v[(2 * i + ii) * 28 + 2 * j + jj];
						}
					}
				}
			}
			return res;
		}
	);
	trainXstr.clear();
	trainYstr.clear();

	std::vector<std::shared_ptr<Scaler<VectorXf>>> scalers;
	scalers.emplace_back(new DummyScaler<VectorXf>());

	std::vector<std::shared_ptr<KernelFunction>> kernels;
	kernels.emplace_back(new QuarticKernel());

	std::vector<std::shared_ptr<DistanceFunction<VectorXf>>> distances;
	distances.emplace_back(new CosineDistance());

		for(size_t k = 12; k <= 12; ++k) {
			for(const auto& distance: distances) {
				for(const auto& kernel: kernels) {
					KNNClassifier<VectorXf, Label> knn(k, distance, kernel);
					knn.train(trainX, trainY);
					auto prediction = knn.predict(test);
					cout << "Id,Prediction" << "\n";
					for(size_t i = 0; i < prediction.rowsNumber(); ++i) {
						cout << i + 1 << "," << prediction[i] << "\n";
					}
				}
			}
		}

	return 0;
}
