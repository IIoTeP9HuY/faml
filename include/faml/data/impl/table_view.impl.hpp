#ifndef TABLE_VIEW_IMPL_HPP
#define TABLE_VIEW_IMPL_HPP

#include <vector>
#include <unordered_map>
#include <utility>

namespace faml {

template <typename RowType>
Table<RowType> TableView<RowType>::toTable() const {
	Table<RowType> result(columnsNames());
	for (const auto &sample : (*this)) {
		result.addRow(sample);
	}
	return result;
}

template <typename RowType>
TableRowsProxy<RowType> TableView<RowType>::operator [] (const std::vector<size_t> &indicies) {
	return TableRowsProxy<RowType>(*this, indicies);
}

template <typename RowType>
bool TableView<RowType>::empty() const {
	return rowsNumber() == 0;
}

template <typename RowType>
template <typename FunctionType, typename NewRowType>
Table<NewRowType> TableView<RowType>::cast(const FunctionType& castFunction) const {
	return cast(castFunction, columnsNames());
}

template <typename RowType>
template <typename FunctionType, typename NewRowType>
Table<NewRowType> TableView<RowType>::cast(const FunctionType& castFunction, const std::vector<std::string>& newColumnsNames) const {
	Table<NewRowType> castedTable(newColumnsNames);
	for (const RowType &sample : (*this)) {
		castedTable.addRow(castFunction(sample));
	}
	return castedTable;
}

template <typename RowType>
template <typename NewRowType, typename FunctionType>
Table<NewRowType> TableView<RowType>::castByElement(FunctionType castFunction) const {
	Table<NewRowType> castedTable(columnsNames());
	for (const RowType &sample : (*this)) {
		NewRowType newRow(sample.size());
		for(size_t i = 0; i < sample.size(); ++i) {
			newRow[i] = castFunction(sample[i]);
		}
		castedTable.addRow(newRow);
	}
	return castedTable;
}

template <typename RowType>
std::pair< Table<RowType>, Table<RowType> > TableView<RowType>::splitOnColumns(const std::vector<std::string> &splitColumnsNames) const {
	std::vector<char> columnMatches(columnsNumber(), false);

	std::vector<std::string> columns = columnsNames();
	std::unordered_map<std::string, size_t> columnsNamesIndices;
	for (size_t i = 0; i < columns.size(); ++i) {
		columnsNamesIndices[columns[i]] = i;
	}

	for (size_t i = 0; i < splitColumnsNames.size(); ++i) {
		auto columnIt = columnsNamesIndices.find(splitColumnsNames[i]);
		if (columnIt == columnsNamesIndices.end()) {
			throw std::invalid_argument("Can't split on not existing column \"" + splitColumnsNames[i] + "\"");
		}
		columnMatches[columnIt->second] = true;
	}

	std::vector<std::string> notMatchedColumnsNames;
	std::vector<std::string> matchedColumnsNames;
	for (size_t i = 0; i < columnsNumber(); ++i) {
		if (!columnMatches[i]) {
			notMatchedColumnsNames.push_back(columns[i]);
		} else {
			matchedColumnsNames.push_back(columns[i]);
		}
	}

	Table<RowType> notMatchedTable(notMatchedColumnsNames);
	Table<RowType> matchedTable(matchedColumnsNames);

	for (size_t row = 0; row < rowsNumber(); ++row) {
		RowType notMatchedRowColumns(notMatchedTable.columnsNumber());
		RowType matchedRowColumns(matchedTable.columnsNumber());
		size_t notMatchedRowColumnsNumber = 0;
		size_t matchedRowColumnsNumber = 0;

		for (size_t column = 0; column < columnsNumber(); ++column) {
			if (!columnMatches[column]) {
				notMatchedRowColumns[notMatchedRowColumnsNumber++] = (*this)[row][column];
			} else {
				matchedRowColumns[matchedRowColumnsNumber++] = (*this)[row][column];
			}
		}

		notMatchedTable.addRow(notMatchedRowColumns);
		matchedTable.addRow(matchedRowColumns);
	}

	return std::make_pair(notMatchedTable, matchedTable);
}

} // namespace faml

#endif // TABLE_VIEW_IMPL_HPP
