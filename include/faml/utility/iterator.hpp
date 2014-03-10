#ifndef ITERATOR_HPP
#define ITERATOR_HPP

namespace faml {

template<typename BaseIterator>
class Iterator {
public:
	typedef typename BaseIterator::value_type value_type;
	Iterator(std::unique_ptr<BaseIterator>&& base): base(std::move(base)) {
	}

	virtual Iterator& operator ++ () {
		++*base;
		return *this;
	}

	virtual Iterator& operator -- () {
		--*base;
		return *this;
	}

	virtual const value_type operator * () const {
		return **base;
	}
	virtual bool operator == (const Iterator& rhs) const {
		return *base == *rhs.base;
	}

	virtual bool operator != (const Iterator& rhs) const {
		return !(*this == rhs);
	}
private:
	std::unique_ptr<BaseIterator> base;
};
} // namesapce faml
#endif // ITERATOR_HPP
