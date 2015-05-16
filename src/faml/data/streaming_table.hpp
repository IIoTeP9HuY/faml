#pragma once

#include "faml/utility/iterator.hpp"

namespace faml {

template<typename RowType>
class StreamingTable {
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
		typedef RowType value_type;
	};
	typedef Iterator<BaseIterator> iterator;

	StreamingTable() {}

	virtual ~StreamingTable() {}

	virtual iterator begin() const = 0;

	virtual iterator end() const = 0;
};

} // namespace faml
