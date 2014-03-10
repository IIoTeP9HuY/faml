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

	auto rtrainX = trainXstr.castByElement<VectorXf>(
		[](const std::string& x) {
			return std::stod(x);
		}
	);
	cerr << "casted X" << endl;
	trainXstr.clear();
	trainYstr.clear();
	cerr << "before preprocessing" << endl;
	int size = 28;


	/*auto trainX = rtrainX.cast(
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
	 
	size /= 2;*/

	auto trainX = rtrainX.cast(
		[size](const VectorXf& vv) {
			VectorXf res = VectorXf::Zero(size * size);

			for(int i = 0; i < size; ++i) {
				for(int j = 0; j < size; ++j) {
					int cnt = 0;
					for(int ii = -1; ii <= 1; ++ii) {
						for(int jj = -1; jj <= 1; ++jj){
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

	rtrainX.clear();
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
	//kernels.emplace_back(new DiscreteKernel());
	//kernels.emplace_back(new InverseKernel());
	//kernels.emplace_back(new RBFKernel(1.0));
	//kernels.emplace_back(new EpanechnikovKernel());

	std::vector<std::unique_ptr<DistanceFunction<VectorXf>>> distances;
	//distances.emplace_back(new EuclidianDistance());
	//distances.emplace_back(new MinkowskiDistance(3));
	//distances.emplace_back(new MinkowskiDistance(5));
	distances.emplace_back(new CosineDistance());
	//distances.emplace_back(new OverlapDistance());

//	for(const auto& scaler: scalers) {
//		scaler->train(subtrainX);
//		auto lambda = [&scaler](const VectorXf& row) {
//			return (*scaler)(row);
//		};
//		auto scaledX = subtrainX.cast(lambda);
//		auto scaledTest = subtestX.cast(lambda);
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
//	}

	return 0;
}
