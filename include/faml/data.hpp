#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <utility>

namespace faml {

template<typename RowType>
class Table {
	typedef std::vector<RowType> DataContainer;
	typedef typename DataContainer::iterator iterator;
	typedef typename DataContainer::const_iterator const_iterator;
public:

	Table() {}

	Table(const std::vector<std::string> &columnsNames): columnsNames(columnsNames) {
		for (size_t i = 0; i < columnsNumber(); ++i) {
			columnsNamesIndices[columnsNames[i]] = i;
		}
	}

	void addRow(const RowType &sample) {
		data.push_back(sample);
	}

	void resizeRows(size_t size) {
		data.resize(size);
	}

	RowType &operator [] (size_t index) {
		return data[index];
	}

	const RowType &operator [] (size_t index) const {
		return data[index];
	}

	iterator begin() {
		return data.begin();
	}

	const_iterator begin() const {
		return data.cbegin();
	}

	iterator end() {
		return data.end();
	}

	const_iterator end() const {
		return data.cend();
	}

	size_t columnsNumber() const {
		return columnsNames.size();
	}

	size_t rowsNumber() const {
		return data.size();
	}

	bool empty() const {
		return rowsNumber() == 0;
	}

	template <typename FunctionType, typename NewRowType = decltype(std::declval<FunctionType>()(std::declval<RowType>()))>
	Table<NewRowType> cast(FunctionType castFunction) const {
		Table<NewRowType> castedTable(columnsNames);
		for (const RowType &sample : data) {
			castedTable.addRow(castFunction(sample));
		}
		return castedTable;
	}

	template <typename NewRowType, typename FunctionType>
	Table<NewRowType> castByElement(FunctionType castFunction) const {
		Table<NewRowType> castedTable(columnsNames);
		for (const RowType &sample : data) {
			NewRowType newRow(sample.size());
			for(size_t i = 0; i < sample.size(); ++i)
				newRow[i] = castFunction(sample[i]);
			castedTable.addRow(newRow);
		}
		return castedTable;
	}

	std::pair< Table<RowType>, Table<RowType> > splitOnColumns(const std::vector<std::string> &splitColumnsNames) const {
		std::vector<char> columnMatches(columnsNumber(), false);
		std::vector<size_t> splitColumnsIndices;
		for (size_t i = 0; i < splitColumnsNames.size(); ++i) {
			auto columnIt = columnsNamesIndices.find(splitColumnsNames[i]);
			if (columnIt == columnsNamesIndices.end()) {
				throw std::invalid_argument("Can't split on not existing column \"" + splitColumnsNames[i] + "\"");
			}
			splitColumnsIndices.push_back(columnIt->second);
			columnMatches[columnIt->second] = true;
		}

		std::vector<std::string> notMatchedColumnsNames;
		std::vector<std::string> matchedColumnsNames;
		for (size_t i = 0; i < columnsNumber(); ++i) {
			if (!columnMatches[i]) {
				notMatchedColumnsNames.push_back(columnsNames[i]);
			} else {
				matchedColumnsNames.push_back(columnsNames[i]);
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
					notMatchedRowColumns[notMatchedRowColumnsNumber++] = data[row][column];
				} else {
					matchedRowColumns[matchedRowColumnsNumber++] = data[row][column];
				}
			}

			notMatchedTable.addRow(notMatchedRowColumns);
			matchedTable.addRow(matchedRowColumns);
		}

		return std::make_pair(notMatchedTable, matchedTable);
	}

	const std::vector<std::string>& getColumnsNames() const {
		return columnsNames;
	}

private:
	DataContainer data;
	std::vector<std::string> columnsNames;
	std::unordered_map<std::string, size_t> columnsNamesIndices;
};

} // namespace faml

#endif // DATA_HPP
