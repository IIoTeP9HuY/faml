#include "faml/io.hpp"
#include "faml/utility/utility.hpp"
#include "faml/models/tree.hpp"
#include "faml/models/tree/trainers/ID3PruningTrainer.hpp"
#include "faml/statistics/informativity_criteria.hpp"
#include "faml/cross_validation/cross_validation.hpp"
#include "faml/cross_validation/shuffle_split.hpp"
#include "faml/quality/classification.hpp"
#include <vector>

#include <iostream>
using namespace std;
using namespace faml;
int main(int argc, char** argv) {
	if(argc < 3) {
		cerr << "usage: " << argv[0] << " train test";
		exit(1);
	}

	auto data = readCSV(argv[1]);
	auto test = readCSV(argv[2]);
	Table<vector<string>> x, _y;
	std::tie(x, _y) = data.splitOnColumns({"50k"});
	auto y = _y.cast(firstElement<vector<string>>);
	typedef vector<string> Row;
	typedef string Label;
	auto predictor = std::make_shared<TreeClassifier<ID3PruningTrainer<Row, Label>>>(ID3PruningTrainer<Row, Label>(std::make_shared<EntropyCriteria<Label>>(), 0.7, 42));
	predictor->train(x, y);
	auto prediction = predictor->predict(test);
	cout << "Id,Solution\n";
	for(size_t i = 0; i < prediction.rowsNumber(); ++i) {
		cout << (i + 1) << "," << prediction[i] << "\n";
	}
	return 0;
}
