#include "faml/io.hpp"
#include "faml/models/logistic_regression.hpp"
#include "faml/preprocessing/scaler.hpp"
#include "faml/quality/classification.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>

#include <iostream>

using namespace faml;
using namespace Eigen;
using namespace std;

int main(int argc, char** argv)
{
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " dataset" << endl;
		return 0;
	}
	string datasetFilename(argv[1]);

	cerr << "Reading dataset \"" << datasetFilename << "\"" << endl;
	Table<vector<string>> dataset(readCSV(datasetFilename));
	cerr << "Rows: " << dataset.rowsNumber() << ", Columns: " << dataset.columnsNumber() << endl;

	Table< std::vector<std::string> > trainSamplesString;
	Table< std::vector<std::string> > trainLabelsString;
	std::tie(trainSamplesString, trainLabelsString) = dataset.splitOnColumns({"click"});

	const int hashSpaceSize = 100000;

	cerr << "Transforming dataset" << endl;

	using DataType = SparseVector<float>;
	Table<DataType> trainSamples(trainSamplesString.cast(
								 [](const std::vector<std::string> &sample) {
								 		auto hasher = std::hash<string>();
								 		auto features = DataType(hashSpaceSize);
										for (size_t i = 0; i < sample.size(); ++i) {
											auto string_feature = to_string(i) + "_" + sample[i];
											int feature = hasher(string_feature) % hashSpaceSize;
											features.coeffRef(feature) = 1;
										}
										return features;
									}
								 ));

	Table<double> trainLabels(trainLabelsString.cast(
								 [](const std::vector<std::string> &sample) {
										return std::stod(sample[0]);
									}
								 ));

	cerr << "Training logistic regression" << endl;
	LogisticRegressor<DataType> regressor(hashSpaceSize, 100);
	regressor.train(trainSamples, trainLabels);

	auto predictedLabels = regressor.predict(trainSamples);
	double loss = logLossScore(trainLabels, predictedLabels);
	std::cerr << "Loss: " << loss << endl;

	return 0;
}
