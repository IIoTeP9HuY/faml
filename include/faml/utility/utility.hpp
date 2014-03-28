#ifndef UTILITY_HPP
#define UTILITY_HPP

namespace faml {

template<typename T, typename U> 
struct bySecond {
	bool operator (const std::pair<T, U>& lhs, const std::pair<T, U>& rhs) const {
		return lhs.second < rhs.second;
	}
};

template <typename Row>
Row majorantClass(const TableView<Row>& y) {
	if(y.rowsNumber() == 0) {
		throw invalid_argument("Empty table");
	}
	map<Row, size_t> counter;
	for(const auto& row: y) {
		++counter[row];
	}
	return max_element(counter.begin(), counter.end(), bySecond<Row, size_t>())->first;
}

} // faml

#endif
