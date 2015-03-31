#include "faml/io.hpp"
#include "faml/models/linear_regression.hpp"
#include "faml/preprocessing/scaler.hpp"
#include "faml/quality/regression.hpp"

#include <eigen3/Eigen/Core>

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
	Table<vector<string>> dataset(readCSV(datasetFilename, false));
	cerr << "Rows: " << dataset.rowsNumber() << ", Columns: " << dataset.columnsNumber() << endl;

	Table< std::vector<std::string> > trainSamplesString;
	Table< std::vector<std::string> > trainLabelsString;
	std::tie(trainSamplesString, trainLabelsString) = dataset.splitOnColumns({"8"});

	Table<VectorXf> trainSamples(trainSamplesString.castByElement<VectorXf>(
								 [](const std::string &sample) {
										return std::stof(sample);
									}
								 ));

	Table<double> trainLabels(trainLabelsString.cast(
								 [](const std::vector<std::string> &sample) {
										return std::stod(sample[0]);
									}
								 ));

	MinMaxScaler scaler(trainSamples.columnsNumber(), 0, 1);
	scaler.train(trainSamples);

	trainSamples = scaler(trainSamples);

	LinearRegressor regressor;
	regressor.train(trainSamples, trainLabels);

	auto predictedLabels = regressor.predict(trainSamples);
	double loss = rmseLoss(trainLabels, predictedLabels);
	std::cerr << "Loss: " << loss << endl;

	return 0;
}
