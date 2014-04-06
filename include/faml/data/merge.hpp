#ifndef TABLE_MERGE_HPP
#define TABLE_MERGE_HPP
#include "faml/data/table.hpp"
#include <stdexcept>
namespace faml {
namespace impl {
	template <typename T>
	std::vector<T> merge(std::vector<T> lhs, const std::vector<T>& rhs) {
		lhs.insert(lhs.end(), rhs.begin(), rhs.end());
		return lhs;
	}
} //namespace impl

template <typename T>
Table<T> merge(const TableView<T>& lhs, const TableView<T>& rhs) {
	if(lhs.rowsNumber() != rhs.rowsNumber())
		throw std::invalid_argument("merge(Table, Table): different sizes");
	Table<T> result(impl::merge(lhs.columnsNames(), rhs.columnsNames()));
	for(size_t i = 0; i < lhs.rowsNumber(); ++i) {
		result.addRow(impl::merge(lhs[i], rhs[i]));
	}
	return result;
}

} //namespace faml

#endif //TABLE_MERGE_HPP
