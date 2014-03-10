#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <iomanip>
#include <tuple>
#include <memory>

#include <eigen3/Eigen/LU>

#include "faml/data/table.hpp"
#include "faml/kernels.hpp"
#include "faml/distances.hpp"
#include "faml/io.hpp"
#include "faml/models/knn.hpp"
#include "faml/preprocessing/scaler.hpp"
#include "faml/quality/classification.hpp"
#include "faml/statistics/statistics.hpp"

using namespace std;
using namespace faml;
using namespace Eigen;

const bool VERBOSE = false;

std::mt19937 gen(43);

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


/*
template<typename DataType, typename LabelType>
void removeOutliers(Table<DataType> *samples,
					Table<LabelType> *labels, DistanceFunction *distance, double dropRate) {

	size_t samplesNumber = samples->size();

	std::vector<double> minDistanceToOtherClass(samplesNumber, std::numeric_limits<double>::max());
	std::vector<double> minDistanceToNativeClass(samplesNumber, std::numeric_limits<double>::max());

	double maxOtherDistance = std::numeric_limits<double>::min();
	double maxNativeDistance = std::numeric_limits<double>::min();

	for (size_t i = 0; i < samplesNumber; ++i) {
		for (size_t j = i + 1; j < samplesNumber; ++j) {

			double dist = (*distance)(samples->at(i), samples->at(j));

			if (labels->at(i) != labels->at(j)) {
				minDistanceToOtherClass[i] = std::min(minDistanceToOtherClass[i], dist);
				minDistanceToOtherClass[j] = std::min(minDistanceToOtherClass[j], dist);
				maxOtherDistance = std::max(maxOtherDistance, dist);
			} else {
				minDistanceToNativeClass[i] = std::min(minDistanceToNativeClass[i], dist);
				minDistanceToNativeClass[j] = std::min(minDistanceToNativeClass[j], dist);
				maxNativeDistance = std::max(maxNativeDistance, dist);
			}

		}
	}

	int otherBinsNumber = std::min(10000, int(maxOtherDistance));
	double otherBinSize = maxOtherDistance / otherBinsNumber;
	std::vector<double> otherDistBins(otherBinsNumber);
	for (size_t i = 0; i < samplesNumber; ++i) {
		++otherDistBins[int(minDistanceToOtherClass[i] / maxOtherDistance * otherBinsNumber)];
	}

	int nativeBinsNumber = std::min(10000, int(maxNativeDistance));
	double nativeBinSize = maxNativeDistance / nativeBinsNumber;
	std::vector<double> nativeDistBins(nativeBinsNumber);
	for (size_t i = 0; i < samplesNumber; ++i) {
		++nativeDistBins[int(minDistanceToNativeClass[i] / maxNativeDistance * nativeBinsNumber)];
	}

	std::cerr << "Drop rate: " << dropRate << std::endl;
	std::cerr << "Max drop number: " << dropRate * samplesNumber << std::endl;

	std::cerr << "Other bins number: " << otherBinsNumber << std::endl;
	std::cerr << "Other bin size: " << otherBinSize << std::endl;


	std::cerr << "Native bins number: " << nativeBinsNumber << std::endl;
	std::cerr << "Native bin size: " << nativeBinSize << std::endl;

	int otherLowerBound = 0;
	int nativeUpperBound = maxNativeDistance * 2;

	int currentOtherNumber = 0;
	for (int i = 0; i < otherBinsNumber; ++i) {
		currentOtherNumber += otherDistBins[i];
		if (currentOtherNumber > samplesNumber * dropRate) {
			otherLowerBound = (maxOtherDistance / otherBinsNumber) * (i);
			std::cerr << "Other bin: " << i << std::endl;
			std::cerr << "Other distance outliers: " << currentOtherNumber << std::endl;
			std::cerr << "Other distance lower bound: " << otherLowerBound << std::endl;
			break;	
		}
	}

	int currentNativeNumber = 0;
	for (int i = nativeBinsNumber - 1; i >= 0; --i) {
		currentNativeNumber += nativeDistBins[i];
		if (currentNativeNumber > samplesNumber * dropRate) {
			nativeUpperBound = (maxNativeDistance / nativeBinsNumber) * (i + 1);
			std::cerr << "Native bin: " << i << std::endl;
			std::cerr << "Native distance outliers: " << currentNativeNumber << std::endl;
			std::cerr << "Native distance upper bound: " << nativeUpperBound << std::endl;
			break;	
		}
	}

	std::vector<sample_type> filteredSamples;
	std::vector<label> filteredLabels;

	double outliersNumber = 0;

	for (size_t i = 0; i < samplesNumber; ++i) {
		if ((minDistanceToOtherClass[i] < otherLowerBound) || (minDistanceToNativeClass[i] > nativeUpperBound)) {
			++outliersNumber;
			continue;
		}

		filteredSamples.push_back(samples->at(i));
		filteredLabels.push_back(labels->at(i));
	}
	std::cerr << "Outliers number: " << outliersNumber << std::endl;

	*samples = filteredSamples;
	*labels = filteredLabels;
}
*/

/*
matrix<double> findOptimalWeights(const std::vector<sample_type> &samples, 
									const std::vector<label> &labels, double alpha, int power) {

	size_t featuresDimension = samples[0].size();
	matrix<double> positive(featuresDimension, 1);
	matrix<double> negative(featuresDimension, 1);

	int positiveNumber = 0;
	int negativeNumber = 0;

	for (size_t i = 0; i < samples.size(); ++i) {
		for (size_t j = i + 1; j < samples.size(); ++j) {
			if (labels[i] == labels[j]) {
				for (size_t k = 0; k < featuresDimension; ++k) {
					positive(k) += fastpow(fabs(samples[i](k) - samples[j](k)), power);
				}
				++positiveNumber;
			} else {
				for (size_t k = 0; k < featuresDimension; ++k) {
					negative(k) += fastpow(fabs(samples[i](k) - samples[j](k)), power);
				}
				++negativeNumber;
			}
		}
	}

	positive /= positiveNumber;
	negative /= negativeNumber;

	return -(alpha * positive - (1 - alpha) * negative);
}
*/
//	return covarianceMatrix.inverse();
//}

class Timer {
public:
	Timer(const std::string &name): name(name), startTime(clock()), stopped(false) {
	}

	~Timer() {
		if (!stopped) {
			stop();
		}
	}

	void stop() {
		size_t stopTime = clock();
		double deltaMs = (stopTime - startTime) * 1.0 / CLOCKS_PER_SEC;
		std::cerr << name << " elapsed: " << deltaMs << " ms" << std::endl;
		stopped = true;
	}

private:
	std::string name;
	size_t startTime;
	bool stopped;
};

template<typename LabelType>
void printPrediction(const std::string &filename, const Table<LabelType>& predictions) {
	std::ofstream outputStream(filename);
	outputStream << "Id,Prediction\n";

	for (size_t i = 0; i < predictions.rowsNumber(); ++i) {
		outputStream << i + 1 << "," << predictions[i] << "\n";
	}
}

typedef VectorXf sampleType;
typedef unsigned long labelType;

int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " train [test]" << endl;
		exit(EXIT_FAILURE);
	}

	std::string trainsetFilename(argv[1]);
	Table< std::vector<std::string> > dataset(readCSV(trainsetFilename));

	Table< std::vector<std::string> > trainSamplesString;
	Table< std::vector<std::string> > trainLabelsString;
	std::tie(trainSamplesString, trainLabelsString) = dataset.splitOnColumns({"label"});

	Table<sampleType> trainSamples(trainSamplesString.castByElement<sampleType>(
								 [](const std::string &sample) {
										return std::stof(sample);
									}
								 ));

	Table<labelType> trainLabels(trainLabelsString.cast(
								 [](const std::vector<std::string> &sample) {
										return std::stoul(sample[0]);
									}
								 ));

	trainSamplesString.clear();
	trainLabelsString.clear();

	Table<sampleType> trainSamplesTrain(trainSamples.columnsNames());
	Table<labelType> trainLabelsTrain(trainLabels.columnsNames());
	Table<sampleType> trainSamplesTest(trainSamples.columnsNames());
	Table<labelType> trainLabelsTest(trainLabels.columnsNames());

	trainTestSplit(trainSamples, trainLabels,
				trainSamplesTrain, trainLabelsTrain,
				trainSamplesTest, trainLabelsTest,
				0.1);

	std::vector<std::unique_ptr<Scaler<sampleType>>> scalers;
	scalers.emplace_back(new DummyScaler<sampleType>());
//	scalers.emplace_back(new PowerAmplifyScaler(0.5));
//	scalers.emplace_back(new NormalScaler());
//	scalers.emplace_back(new MinMaxScaler(trainSamples.columnsNumber(), 0, 1));

	std::vector<std::unique_ptr<DistanceFunction<VectorXf>>> distances;
//	distances.emplace_back(new EuclidianDistance());
//	distances.emplace_back(new MinkowskiDistance(3.0));
//	distances.emplace_back(new MinkowskiDistance(5.0));
	distances.emplace_back(new CosineDistance());
//	distances.emplace_back(new OverlapDistance());

	std::vector<std::unique_ptr<KernelFunction>> kernels;
//	kernels.emplace_back(new RBFKernel(1.0));
//	kernels.emplace_back(new RBFKernel(4.0));
//	kernels.emplace_back(new InverseKernel());
//	kernels.emplace_back(new TriangleKernel());
	kernels.emplace_back(new QuarticKernel());
//	kernels.emplace_back(new EpanechnikovKernel());
//	kernels.emplace_back(new DiscreteKernel());

	for (const auto &scaler : scalers) {
		(*scaler).train(trainSamplesTrain);
		Table<sampleType> trainSamplesTrainScaled = (*scaler)(trainSamplesTrain);
		Table<sampleType> trainSamplesTestScaled = (*scaler)(trainSamplesTest);

		for (const auto &distance : distances) {
			for (const auto &kernel : kernels) {
				for (size_t K = 1; K < 20; ++K) {
					KNNClassifier<sampleType, labelType> classifier(K, *distance, *kernel);
					classifier.train(trainSamplesTrainScaled, trainLabelsTrain);
					Table<labelType> trainLabelsTestPrediction = classifier.predict(trainSamplesTestScaled);

					double accuracy = accuracyScore(trainLabelsTest, trainLabelsTestPrediction);
					std::cout << "Accuracy: " << accuracy << std::endl;
					std::cout << "(K: " << K << "; " << scaler->toString() << "; "
							  << distance->toString() <<  "; " << kernel->toString() << ")" << std::endl << std::endl;
				}
			}
		}
	}

	if (argc == 3) {
		std::string testsetFilename(argv[2]);
		Table<std::vector<std::string>> testSamplesString(readCSV(testsetFilename));
		Table<sampleType> testSamples(testSamplesString.castByElement<sampleType>(
									 [](const std::string &sample) {
											return std::stof(sample);
										}
									 ));
//		for (auto &sample : testSamples) {
//			sample = normalScaler(sample);
//		}

		CosineDistance cosineDistance;
		TriangleKernel triangleKernel;
		KNNClassifier<sampleType, labelType> classifier(12, cosineDistance, triangleKernel);
		classifier.train(trainSamplesTrain, trainLabelsTrain);
		Table<labelType> trainLabelsTestPrediction = classifier.predict(testSamples);
		printPrediction("predictions.csv", trainLabelsTestPrediction);
	}

	return 0;
}
