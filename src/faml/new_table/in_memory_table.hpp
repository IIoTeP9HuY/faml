#pragma once

#include "table.hpp"

#include <vector>

namespace faml {

template<typename RowType>
class InMemoryTable : public Table<RowType> {
public:
	InMemoryTable()
	{ }

	explicit InMemoryTable(std::vector<RowType> data)
		: data(std::move(data))
	{ }

	size_t size() const {
		return data.size();
	}

	const RowType& operator [] (size_t index) const {
		return data.at(index);
	}

private:
	std::vector<RowType> data;
};

} // namespace faml
