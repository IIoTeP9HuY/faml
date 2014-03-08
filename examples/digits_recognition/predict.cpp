#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <iomanip>
#include <tuple>

#include "faml/data.hpp"
#include "faml/kernels.hpp"
#include "faml/distances.hpp"
#include "faml/io.hpp"
#include "faml/models/knn.hpp"
#include "faml/preprocessing/scaler.hpp"

using namespace std;
using namespace faml;
using namespace Eigen;

const bool VERBOSE = false;

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

struct PredictionStatistics {
	PredictionStatistics() {
		accuracy = 0.0;
	}

	double accuracy;

};

template<typename LabelType>
struct ConfustionMatrix {
	ConfustionMatrix(const std::vector<LabelType> &labels): labels(labels) {
	}

	void printConfusionMatrix() {
		std::cout << "  ";
		for (const LabelType &label : labels) {
			std::cout << std::setw(4) << label << " ";
		}
		std::cout << std::endl;

		for (const LabelType &firstLabel : labels) {
			std::cout << firstLabel << " ";
			for (const LabelType &secondLabel : labels) {
				std::cout << std::setw(4) << confusionMatrix[firstLabel][secondLabel] << " ";
			}
			std::cout << endl;
		}
	}

	std::vector<LabelType> labels;
	std::unordered_map< LabelType, std::unordered_map<LabelType, size_t> > confusionMatrix;
};

template<typename LabelType>
PredictionStatistics calculatePredictionStatistics(const Table<LabelType> &realLabels,
												   const Table<LabelType> &predictedLabels) {
	PredictionStatistics statistics;
	int matches = 0;
	size_t N = realLabels.rowsNumber();
	assert(N == predictedLabels.rowsNumber());

	for (size_t i = 0; i < N; ++i) {
		if (realLabels[i] == predictedLabels[i]) {
			++matches;
		}
	}
	statistics.accuracy = matches * 1.0 / N;
	return statistics;
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

template<typename DataType>
MatrixXf buildInverseCovarianceMatrix(const Table<DataType> &samples) {
	size_t featuresDimension = samples.columnsNumber();

	std::cerr << "Features dimennsion: " << featuresDimension << std::endl;

	MatrixXf samplesMatrix(samples.size(), featuresDimension);
	for (size_t i = 0; i < samples.size(); ++i) {
		samplesMatrix(i) = samples[i];
	}

	MatrixXf means = samplesMatrix.colwise().sum();
	for (size_t i = 0; i < featuresDimension; ++i) {
		samplesMatrix.col(i) -= means(i);
	}

	MatrixXf covarianceMatrix(featuresDimension, featuresDimension);
	for (size_t i = 0; i < featuresDimension; ++i) {
		for (size_t j = 0; j < featuresDimension; ++j) {
			covarianceMatrix(i, j) = samplesMatrix.col(i).transpose() * samplesMatrix.col(j);
		}
	}

	return covarianceMatrix.inverse();
}

//void printPrediction(const std::string &filename, const std::vector<label>& predictions) {
//	std::ofstream outputStream(filename);
//	outputStream << "Id,Prediction\n";

//	for (size_t i = 0; i < predictions.size(); ++i) {
//		outputStream << i + 1 << "," << predictions[i] << "\n";
//	}
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

typedef VectorXf sampleType;
typedef size_t labelType;

int main(int argc, char **argv) {
	Timer predictTimer("predict");

	if(argc < 2) {
		cerr << "usage: " << argv[0] << " train [test]" << endl;
		exit(EXIT_FAILURE);
	}

	Timer prepareTimer("prepare");

	std::string trainsetFilename(argv[1]);
	Table< std::vector<std::string> > dataset(readCSV(trainsetFilename));

//	MatrixXf inverseCovarianceMatrix;
//	inverseCovarianceMatrix = buildInverseCovarianceMatrix(trainSamples);

	Table< std::vector<std::string> > trainSamplesString;
	Table< std::vector<std::string> > trainLabelsString;
	std::tie(trainSamplesString, trainLabelsString) = dataset.splitOnColumns({"Number"});

	Table<sampleType> trainSamples(trainSamplesString.cast<sampleType>(
								 [](const std::vector<std::string> &sample) {
										sampleType result(sample.size());
										for (size_t i = 0; i < sample.size(); ++i)
											result[i] = std::atof(sample[i].c_str());
										return result;
									}
								 ));

	Table<labelType> trainLabels(trainLabelsString.cast<labelType>(
								 [](const std::vector<std::string> &sample) {
										return labelType(std::atol(sample[0].c_str()));
									}
								 ));

	Table<sampleType> trainSamplesTrain(trainSamples.getColumnsNames());
	Table<labelType> trainLabelsTrain(trainLabels.getColumnsNames());
	Table<sampleType> trainSamplesTest(trainSamples.getColumnsNames());
	Table<labelType> trainLabelsTest(trainLabels.getColumnsNames());

	trainTestSplit(trainSamples, trainLabels,
				trainSamplesTrain, trainLabelsTrain,
				trainSamplesTest, trainLabelsTest,
				0.1);

	NormalScaler normalScaler;
	normalScaler.train(trainSamplesTrain);

	for (auto &sample : trainSamplesTrain) {
		sample = normalScaler(sample);
	}

	for (auto &sample : trainSamplesTest) {
		sample = normalScaler(sample);
	}

	int power = 2;

	MinkowskiDistance<sampleType> minkowskiDistance(power);
    double gamma = 4.0;
	RBFKernel rbfKernel(gamma);
	InverseKernel inverseKernel();
    TriangleKernel triangleKernel;
    QuarticKernel quarticKernel;
    EpanechnikovKernel epanechnikovKernel;
    DiscreteKernel discreteKernel;

    // removeOutliers(&trainSamplesTrain, &trainLabelsTrain, &minkowskiDistance, 0.001);
    // removeOutliers(&trainSamples, &trainLabels, &minkowskiDistance, 0.001);

	MatrixXf weights = VectorXf::Ones(trainSamples.columnsNumber());
    // weights = findOptimalWeights(trainSamplesTrain, trainLabelsTrain, 0.1, power);
    // std::cerr << weights << std::endl;

	WMinkowskiDistance<sampleType> wminkowskiDistance(power, weights);
	// MulticlassWMinkowskiDistance multiclassWminkowskiDistance(power, multiclassWeights);
	// MahalanobisDistance mahalanobisDistance(inverseCovarianceMatrix);
	CosineDistance<sampleType> cosineDistance;
	OverlapDistance<sampleType> overlapDistance;

	prepareTimer.stop();

	{
		Timer knnRunsTimer("knn runs");
		for (size_t K = 1; K < 20; ++K) {
			KNNClassifier<sampleType, labelType> classifier(K, minkowskiDistance, quarticKernel);
			classifier.train(trainSamplesTrain, trainLabelsTrain);
			Table<labelType> trainLabelsTestPrediction = classifier.predict(trainSamplesTest);

			PredictionStatistics statistics = calculatePredictionStatistics(trainLabelsTest, trainLabelsTestPrediction);
			std::cout << "K: " << K << " Accuracy: " << statistics.accuracy << std::endl;
		}
	}

/*
	if (argc == 3) {
		std::string testsetFilename(argv[2]);
		Table<sampleType> testSamples(readCSV<sampleType>(testsetFilename));
		for (auto &sample : testSamples) {
			sample = normalScaler(sample);
		}

		KNNClassifier<sampleType, labelType> classifier(17, minkowskiDistance, rbfKernel);
		classifier.train(trainSamples, trainLabels);
		Table<labelType> trainLabelsTestPrediction = classifier.predict(testSamples);
//		printPrediction("predictions.csv", trainLabelsTestPrediction);
	}
*/

	return 0;
}
