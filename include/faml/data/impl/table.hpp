#ifndef IMPL_TABLE_HPP
#define IMPL_TABLE_HPP

#include <vector>

namespace faml {

template<typename RowType>
class Table : public TableView<RowType> {
	typedef std::vector<RowType> DataContainer;
	typedef TableView<RowType> BaseTable;
	typedef typename BaseTable::BaseIterator ParentBaseIterator;
	typedef typename BaseTable::iterator iterator;
public:

	class BaseIterator : public ParentBaseIterator {
	public:
		BaseIterator(typename DataContainer::const_iterator it): it(it) {}

		virtual BaseIterator& operator ++ () {
			++it;
			return *this;
		}

		virtual BaseIterator& operator -- () {
			--it;
			return *this;
		}

		virtual const RowType& operator * () const {
			return *it;
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
		typename DataContainer::const_iterator it;
	};

	Table() {}

	Table(const std::vector<std::string> &columnsNames): columnsNames_(columnsNames) {
	}

	void addRow(const RowType &sample) {
		data.push_back(sample);
	}

	void addRow(RowType &&sample) {
		data.push_back(std::move(sample));
	}

	void resizeRows(size_t size) {
		data.resize(size);
	}

	using BaseTable::operator [];

	const RowType &operator [] (size_t index) const {
		return data[index];
	}

	iterator begin() const {
		return iterator(std::unique_ptr<ParentBaseIterator>(new BaseIterator(data.cbegin())));
	}

	iterator end() const {
		return iterator(std::unique_ptr<ParentBaseIterator>(new BaseIterator(data.cend())));
	}

	size_t columnsNumber() const {
		return columnsNames_.size();
	}

	size_t rowsNumber() const {
		return data.size();
	}

	void clear() {
		DataContainer empty;
		data.swap(empty);
	}

	virtual const std::vector<std::string>& columnsNames() const {
		return columnsNames_;
	}
private:
	DataContainer data;
	std::vector<std::string> columnsNames_;
};

} // namespace faml


#endif // IMPL_TABLE_HPP
