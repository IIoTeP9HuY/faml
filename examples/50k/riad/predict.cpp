#include "faml/io.hpp"
#include "faml/utility/utility.hpp"
#include "faml/models/tree.hpp"
#include "faml/models/tree/trainers/ID3Trainer.hpp"
#include "faml/statistics/informativity_criteria.hpp"
#include <vector>

#include <iostream>
using namespace std;
using namespace faml;
int main(int argc, char** argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " filename";
		exit(1);
	}

	string file = argv[1];
	auto data = readCSV(file);
	Table<vector<string>> x, _y;
	std::tie(x, _y) = data.splitOnColumns({"50k"});
	auto y = _y.cast(firstElement<vector<string>>);
	typedef vector<string> Row;
	typedef string Label;
	auto predictor = TreeClassifier<ID3Trainer<Row, Label>>(ID3Trainer<Row, Label>(std::make_shared<EntropyCriteria<Label>>()));

	predictor.train(x, y);

	return 0;
}
