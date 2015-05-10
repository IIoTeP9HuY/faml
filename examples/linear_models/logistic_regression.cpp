#include "faml/io.hpp"
#include "faml/models/logistic_regression.hpp"
#include "faml/preprocessing/scaler.hpp"
#include "faml/quality/classification.hpp"
#include "faml/algebra/sparse_vector.hpp"
#include "faml/data/file_streaming_table.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>

#include <iostream>

using namespace faml;
using namespace Eigen;
using namespace std;

//using DataType = Eigen::SparseVector<float>;
using DataType = faml::SparseVector<float>;
using LabelType = double;

int main(int argc, char** argv)
{
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " dataset" << endl;
		return 0;
	}
	string datasetFilename(argv[1]);

	string labelColumn = "click";
	const int hashSpaceSize = 100000;

	cerr << "Reading dataset \"" << datasetFilename << "\"" << endl;
	FileStreamingTable<
		std::pair<DataType, LabelType>,
		FileReader,
		CSVToSparseParserFactory> dataset(
			datasetFilename,
			CSVToSparseParserFactory(labelColumn, hashSpaceSize));

	// TODO(acid) Print some stats about file here (size, etc)
	//Table<vector<string>> dataset(readCSV(datasetFilename));
	//cerr << "Rows: " << dataset.rowsNumber() << ", Columns: " << dataset.columnsNumber() << endl;
	
	cerr << "Training logistic regression" << endl;
	LogisticRegressor<DataType> regressor(hashSpaceSize, 100);
	regressor.train(dataset);

	// TODO(acid) Replace this with score function that won't save predictions into memory
	// --------
	//auto predictedLabels = regressor.predict(trainSamplesAndLabels);
	//double loss = logLossScore(trainLabels, predictedLabels);
	//std::cerr << "Final train loss: " << loss << endl;
	// --------

	return 0;
}
