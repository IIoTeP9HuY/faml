#ifndef TABLE_ROWS_PROXY_HPP
#define TABLE_ROWS_PROXY_HPP

#include <string>
#include <vector>

namespace faml {

template<typename RowType>
class TableRowsProxy : public TableView<RowType> {
	typedef TableView<RowType> BaseTable;
	typedef typename BaseTable::BaseIterator ParentBaseIterator;
	typedef typename BaseTable::iterator iterator;
public:

	class BaseIterator : public ParentBaseIterator {
	public:
		BaseIterator(const TableView<RowType> &table,
				 std::vector<size_t>::const_iterator it): table(table), it(it) {}

		virtual BaseIterator& operator ++ () {
			++it;
			return *this;
		}

		virtual BaseIterator& operator -- () {
			--it;
			return *this;
		}

		virtual const RowType& operator * () const {
			return table[*it];
		}

		virtual bool operator == (const ParentBaseIterator& rhs) const {
			try {
				const BaseIterator &r = dynamic_cast<const BaseIterator&> (rhs);
				return it == r.it;
			} catch (std::bad_cast&) {
				return false;
			}
		}
	private:
		const TableView<RowType> &table;
		std::vector<size_t>::const_iterator it;
	};

	TableRowsProxy() {}

	TableRowsProxy(const TableView<RowType> &table, const std::vector<size_t> &indices):
		table(table), indices(indices) {
	}

	const RowType &operator [] (size_t index) const {
		return table[indices[index]];
	}

	iterator begin() const {
		return iterator(std::unique_ptr<ParentBaseIterator>(new BaseIterator(table, indices.cbegin())));
	}

	iterator end() const {
		return iterator(std::unique_ptr<ParentBaseIterator>(new BaseIterator(table, indices.cend())));
	}

	size_t columnsNumber() const {
		return table.columnsNumber();
	}

	size_t rowsNumber() const {
		return indices.size();
	}

	void clear() {
		indices.clear();
	}

	const std::vector<std::string>& columnsNames() const {
		return table.columnsNames();
	}

private:
	const TableView<RowType> &table;
	std::vector<size_t> indices;
};

} // namespace faml

#endif // TABLE_ROWS_PROXY_HPP
