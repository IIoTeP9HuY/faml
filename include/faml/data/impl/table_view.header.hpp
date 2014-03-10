#ifndef TABLE_VIEW_HEADER_HPP
#define TABLE_VIEW_HEADER_HPP
#include <vector>
#include <string>

namespace faml {

template <typename RowType>
class TableView {
public:

	class BaseIterator {
	public:
		virtual BaseIterator& operator ++ () = 0;
		virtual BaseIterator& operator -- () = 0;
		virtual const RowType& operator * () const = 0;
		virtual bool operator == (const BaseIterator&) const = 0;

		virtual bool operator != (const BaseIterator& rhs) const {
			return !(*this == rhs);
		}

		virtual ~BaseIterator(){}

	};

	class iterator {
	public:
		iterator(std::unique_ptr<BaseIterator>&& base): base(std::move(base)) {
		}

		virtual iterator& operator ++ () {
			++*base;
			return *this;
		}

		virtual iterator& operator -- () {
			--*base;
			return *this;
		}

		virtual const RowType& operator * () const {
			return **base;
		}
		virtual bool operator == (const iterator& rhs) const {
			return *base == *rhs.base;
		}

		virtual bool operator != (const iterator& rhs) const {
			return !(*this == rhs);
		}
	private:
		std::unique_ptr<BaseIterator> base;
	};

	TableView() {}

	virtual ~TableView() {}

	virtual Table<RowType> toTable() const;

	virtual const RowType &operator [] (size_t index) const = 0;

	virtual TableRowsProxy<RowType> operator [] (const std::vector<size_t> &indicies);

	virtual iterator begin() const = 0;

	virtual iterator end() const = 0;

	virtual size_t columnsNumber() const = 0;

	virtual size_t rowsNumber() const = 0;

	virtual bool empty() const;

	template <typename FunctionType, typename NewRowType = decltype(std::declval<FunctionType>()(std::declval<RowType>()))>
	Table<NewRowType> cast(const FunctionType& castFunction) const;

	template <typename FunctionType, typename NewRowType = decltype(std::declval<FunctionType>()(std::declval<RowType>()))>
	Table<NewRowType> cast(const FunctionType& castFunction, const std::vector<std::string>&) const;

	template <typename NewRowType, typename FunctionType>
	Table<NewRowType> castByElement(FunctionType castFunction) const;

	std::pair< Table<RowType>, Table<RowType> > splitOnColumns(const std::vector<std::string> &splitColumnsNames) const;

	virtual const std::vector<std::string>& columnsNames() const  = 0;
};

} // namespace faml

#endif // TABLE_VIEW_HEADER_HPP
