#pragma once

#include <memory>

namespace faml {

template<typename RowType>
class Table;

template<typename OldRowType, typename RowType, typename Function>
class FilteredTable : public Table<RowType> {
public:
	FilteredTable(std::shared_ptr<Table<OldRowType>> table, const Function& func)
		: table(table)
		, func(func)
	{ }

private:
	std::shared_ptr<Table<OldRowType>> table;
	Function func;
};

template<typename RowType>
class Table : public std::enable_shared_from_this<Table<RowType>> {
public:
	template<typename NewRowType, typename Function>
	std::shared_ptr<FilteredTable<RowType, NewRowType, Function>> map(const Function& func) {
		return std::make_shared<FilteredTable<RowType, NewRowType, Function>> (this->shared_from_this(), func);
	}
};

} // namespace faml
