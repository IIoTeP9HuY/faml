#ifndef TABLE_HPP
#define TABLE_HPP

namespace faml {

template<typename RowType>
class TableView;

template<typename RowType>
class Table;

template<typename RowType>
class TableRowsProxy;

} // namespace faml

#include "impl/table_view.header.hpp"
#include "impl/table.hpp"
#include "impl/table_rows_proxy.hpp"

#include "impl/table_view.impl.hpp"

#endif // TABLE_HPP
