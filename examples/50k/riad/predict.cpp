#include "faml/io.hpp"
#include "faml/utility/utility.hpp"
#include <vector>

using namespace std;
using namespace faml;
int main(int argc, char** argv) {
	if(argc < 2) {
		cerr << "usage: " << argv[0] << " filename";
		exit(1);
	}

	string file = argv[1];
	auto data = readCSV(file);
	Table<vector<string>> _x, _y;
	std::tie(x, _y) = data.splitOnColumns({"50k"});
	auto y = _y.cast(firstElement<vector<string>>);

	

	return 0;
}
