#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <iomanip>
#include <tuple>
#include <memory>

#include "faml/data.hpp"
#include "faml/kernels.hpp"
#include "faml/distances.hpp"
#include "faml/io.hpp"
#include "faml/models/knn.hpp"
#include "faml/preprocessing/scaler.hpp"

using namespace std;
using namespace faml;
using namespace Eigen;

std::random_device rd;
std::mt19937 gen(rd());

double uniformUnitRandom() {
	std::uniform_real_distribution<> uniformGenerator(0, 1);
	return uniformGenerator(gen);
}

template<typename DataType, typename LabelType>
void trainTestSplit(const Table<DataType> &samples, const Table<LabelType> &labels,
					Table<DataType> &trainSamples, Table<LabelType> &trainLabels,
					Table<DataType> &testSamples, Table<LabelType> &testLabels,
					double testProportion) {

	size_t N = samples.rowsNumber();

	for (size_t i = 0; i < N; ++i) {
		if (uniformUnitRandom() < testProportion) {
			testSamples.addRow(samples[i]);
			testLabels.addRow(labels[i]);
		} else {
			trainSamples.addRow(samples[i]);
			trainLabels.addRow(labels[i]);
		}
	}
}

int main() {
	auto testData = readCSV("mnist_small_train.csv");
	typedef std::vector<std::string> StrRowType;
	typedef VectorXf SampleType;
	typedef unsigned long long Label;
	Table<StrRowType> trainXstr, trainYstr;
	std::tie(trainXstr, trainYstr) = testData.splitOnColumns({"label"});
	auto trainY = trainYstr.cast(
		[](const StrRowType &sample) { 
			return std::stoull(sample[0]); 
		}
	);

	auto trainX = trainXstr.castByElement<VectorXf>(
		[](const std::string& x) {
			return std::stod(x);
		}
	);
	trainXstr.clear();
	trainYstr.clear();

	auto columns = trainX.getColumnsNames();

	Table<SampleType> subtrainX(columns);
	Table<Label> subtrainY(trainY.getColumnsNames());
	Table<SampleType> subtestX(columns);
	Table<Label> subtestY(trainY.getColumnsNames());

	trainTestSplit(trainX, trainY,
				subtrainX, subtrainY,
				subtestX, subtestY,
				0.05);

	trainX.clear();
	trainY.clear();

	std::vector<std::unique_ptr<Scaler<VectorXf>>> scalers;
	scalers.emplace_back(new DummyScaler<VectorXf>());
	scalers.emplace_back(new NormalScaler());
	scalers.emplace_back(new MinMaxScaler(columns.size(), 0, 1));

	std::vector<std::unique_ptr<KernelFunction>> kernels;
	kernels.emplace_back(new InverseKernel());
	kernels.emplace_back(new DiscreteKernel());
	kernels.emplace_back(new RBFKernel(1.0));
	kernels.emplace_back(new QuarticKernel());
	kernels.emplace_back(new EpanechnikovKernel());

	std::vector<std::unique_ptr<DistanceFunction<VectorXf>>> distances;
	distances.emplace_back(new EuclidianDistance());
	distances.emplace_back(new MinkowskiDistance(3));
	distances.emplace_back(new MinkowskiDistance(5));
	distances.emplace_back(new CosineDistance());
	distances.emplace_back(new OverlapDistance());
	for(const auto& df: distances) {
		cout << df->toString() << endl;
	}
	//distances.emplace_back(new MahalanobisDistance());


	return 0;
}
