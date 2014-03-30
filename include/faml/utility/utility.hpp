#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <stdexcept>
#include <map>
#include <type_traits>
#include <algorithm>

namespace faml {

template<typename T, typename U> 
struct bySecond {
	bool operator ()(const std::pair<T, U>& lhs, const std::pair<T, U>& rhs) const {
		return lhs.second < rhs.second;
	}
};

template <typename Row>
Row majorantClass(const TableView<Row>& y) {
	if(y.rowsNumber() == 0) {
		throw std::invalid_argument("Empty table");
	}
	std::map<Row, size_t> counter;
	for(const auto& row: y) {
		++counter[row];
	}
	return std::max_element(counter.begin(), counter.end(), bySecond<Row, size_t>())->first;
}

template <typename Row>
auto firstElement(const Row& row) -> typename std::decay<decltype(row[0])>::type {
	return row[0];
}

} // namespace faml

#endif // UTILITY_HPP
