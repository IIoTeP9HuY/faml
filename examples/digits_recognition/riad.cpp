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
#include "faml/cross_validation/kfold.hpp"
#include "faml/cross_validation/shuffle_split.hpp"

using namespace std;
using namespace faml;
using namespace Eigen;

std::string print(const vector<size_t>& a) {
	std::stringstream ss;
	ss << "[";
	for(auto x: a)
		ss << x << ' ';
	ss << "]";
	return ss.str();
}
int main() {
	for (const auto& z: ShuffleSplit(14, (size_t)4, 5, 41)) {
		cout << print(z.first) << ' ' << print(z.second) << "\n";
	}
	auto testData = readCSV("mnist_small_train.csv");
	cerr << "read" << endl;
	typedef std::vector<std::string> StrRowType;
	typedef VectorXf SampleType;
	typedef unsigned long long Label;
	Table<StrRowType> trainXstr, trainYstr;
	std::tie(trainXstr, trainYstr) = testData.splitOnColumns({"label"});
	cerr << "splitted" << endl;
	auto trainY = trainYstr.cast(
		[](const StrRowType &sample) { 
			return std::stoull(sample[0]); 
		}
	);
	cerr << "casted Y" << endl;

	auto trainX = trainXstr.castByElement<VectorXf>(
		[](const std::string& x) {
			return std::stod(x);
		}
	);
	cerr << "casted X" << endl;
	trainXstr.clear();
	trainYstr.clear();
	cerr << "before preprocessing" << endl;
	int size = 28;

	trainX = trainX.cast(
		[size](const VectorXf& vv) {
			VectorXf res = VectorXf::Zero(size * size);

			for(int i = 0; i < size; ++i) {
				for(int j = 0; j < size; ++j) {
					int cnt = 0;
					for(int ii = -1; ii <= 1; ++ii) {
						for(int jj = -1; jj <= 1; ++jj){
							if(abs(ii) + abs(jj) == 2)
								continue;
							if(i + ii < 0 || i + ii >= size) {
								continue;
							}
							if(j + jj < 0 || j + jj >= size) {
								continue;
							}
							++cnt;
							res[i * size + j] += vv[(i + ii) * size + j + jj];
						}
					}
					res[i * size + j] /= cnt;
				}
			}
			return res;
		}
	);
	cerr << "preprocessed" << endl;

	auto columns = trainX.columnsNames();

	cerr << "here" << endl;
	auto indicies = trainTestSplit(trainX.rowsNumber(), (size_t)19000, 2013);
	cerr << "here" << endl;
	auto subtrainX = trainX[indicies.first];
	auto subtrainY = trainY[indicies.first];
	auto subtestX = trainX[indicies.second];
	auto subtestY = trainY[indicies.second];


	std::vector<std::unique_ptr<KernelFunction>> kernels;
	kernels.emplace_back(new QuarticKernel());

	std::vector<std::unique_ptr<DistanceFunction<VectorXf>>> distances;
	distances.emplace_back(new CosineDistance());

	for(size_t k = 1; k <= 20; ++k) {
		for(const auto& distance: distances) {
			for(const auto& kernel: kernels) {
				clock_t start = clock();
				KNNClassifier<VectorXf, Label> knn(k, *distance, *kernel);
				double res = crossValidate(knn, subtrainX, subtrainY, KFold(subtrainX.rowsNumber(), 5), AccuracyScorer<Label>());
				cout << k << ' ' <<distance->toString() << ' ' << kernel->toString() << "\n";
				cout << "Score: " << res << "\n";
				cout << "time " << (clock() - start) / 1.0 / CLOCKS_PER_SEC;
				cout << endl;
			}
		}
	}

	return 0;
}
