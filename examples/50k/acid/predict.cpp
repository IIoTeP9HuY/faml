#include <unordered_set>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <iomanip>
#include <tuple>
#include <memory>

#include "faml/io.hpp"
#include "faml/data/table.hpp"
#include "faml/models/tree.hpp"
#include "faml/models/tree/trainers/ID3Trainer.hpp"

using namespace faml;
using namespace std;

int main(int argc, char **argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " train [test]" << endl;
		exit(EXIT_FAILURE);
	}

	string trainsetFilename(argv[1]);
	Table< vector<string> > dataset(readCSV(trainsetFilename));

	Table< vector<string> > trainSamplesString;
	Table< vector<string> > trainLabelsString;
	tie(trainSamplesString, trainLabelsString) = dataset.splitOnColumns({"50k"});

	Table<string> trainLabels(trainLabelsString.cast(
								 [](const vector<string> &sample)->string {
										return sample[0];
									}
								 ));

	typedef vector<string> Row;
	typedef string Label;

	TreeClassifier<ID3Trainer<Row, Label>> classifier{ID3Trainer<Row, Label>(std::make_shared<EntropyCriteria<Label>>())};
	classifier.train(trainSamplesString, trainLabels);

	return 0;
}
