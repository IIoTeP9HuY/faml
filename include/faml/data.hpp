#ifndef DATA_HPP
#define DATA_HPP

#include <vector>
#include <set>
#include <string>

namespace faml {

template<typename RowType>
class Table {
	typedef std::vector<RowType> DataContainer;
	typedef std::vector<std::string> IndexContainer;
public:

	Table(const std::vector<std::string> &index): indexContainer(index) {
	}

	void addRow(const RowType &data) {
		dataContainer.push_back(data);
	}

	RowType &operator [] (size_t index) {
		return dataContainer[index];
	}

	const RowType &operator [] (size_t index) const {
		return dataContainer[index];
	}

	DataContainer::iterator begin() {
		return dataContainer.begin();
	}

	DataContainer::const_iterator begin() const {
		return dataContainer.cbegin();
	}

	DataContainer::iterator end() {
		return dataContainer.end();
	}

	DataContainer::const_iterator end() const {
		return dataContainer.cend();
	}

	size_t rowsNumber() const {
		return dataContainer.size();
	}

	size_t columnsNumber() const {
		return indexContainer.size();
	}

	void splitOnColumns(std::vector<std::string> splitColumns,
						DataContainer &notMatchedColumns,
						DataContainer &matchedColumns) const {
		notMatchedColumns.resize(rowsNumber());
		matchedColumns.resize(rowsNumber());
		std::set<std::string> splitColumnsSet(splitColumns.begin(), splitColumns.end());

		std::vector<char> columnMatches(columnsNumber(), false);
		size_t matchedColumnsNumber = 0;
		for (size_t column = 0; column < columnsNumber(); ++column) {
			if (splitColumnsSet.find(indexContainer[column]) != splitColumnsSet.end()) {
				++matchedColumnsNumber;
				columnMatches[column] = true;
			}
		}

		for (size_t row = 0; row < rowsNumber(); ++row) {
			RowType notMatchedRowColumns(columnsNumber() - matchedColumnsNumber);
			RowType matchedRowColumns(matchedColumnsNumber);
			size_t notMatchedRowColumnsNumber = 0;
			size_t matchedRowColumnsNumber = 0;

			for (size_t column = 0; column < columnsNumber(); ++column) {
				if (!columnMatches[column]) {
					notMatchedRowColumns[notMatchedRowColumnsNumber++] = dataContainer[row][column];
				} else {
					matchedRowColumns[matchedRowColumnsNumber++] = dataContainer[row][column];
				}
			}

			notMatchedColumns.push_back(notMatchedRowColumns);
			matchedColumns.push_back(matchedRowColumnsNumber);
		}
	}

	IndexContainer& indices() {
		return indexContainer;
	}

private:
	DataContainer dataContainer;
	IndexContainer indexContainer;
};

} // namespace faml

#endif // DATA_HPP
