#ifndef KFOLD_HPP
#define KFOLD_HPP

#include <stdexcept>

namespace faml {

struct KFoldIterator {
public:
	KFoldIterator(size_t n, size_t part, size_t parts): n(n), part(part), parts(parts) {
		if(parts == 0 || parts > n) {
			throw std::invalid_argument("Wrong number of parts");
		}
	}
	KFoldIterator& operator ++ () {
		if(part == parts) {
			throw std::invalid_argument("Iterator not incrementable");
		}
		++part;
		return *this;
	}
	KFoldIterator& operator -- () {
		if(part == 0) {
			throw std::invalid_argument("Iterator not decrementable");
		}
		--part;
		return *this;
	}

	bool operator == (const KFoldIterator& rhs) {
		return part == rhs.part && n == rhs.n && parts == rhs.parts;
	}

	bool operator != (const KFoldIterator& rhs) {
		return !(*this == rhs);
	}
	std::pair<std::vector<size_t>, std::vector<size_t>> operator * () const {
		size_t startIndex = index(part);
		size_t endIndex   = index(part + 1);
		size_t len = endIndex - startIndex;
		assert(startIndex < endIndex);
		std::vector<size_t> train, test;
		train.reserve(n - len);
		test.reserve(len);
		for(size_t i = 0; i < startIndex; ++i) {
			train.push_back(i);
		}
		for(size_t i = startIndex; i < endIndex; ++i) {
			test.push_back(i);
		}
		for(size_t i = endIndex; i < n; ++i) {
			train.push_back(i);
		}
		return std::make_pair(train, test);
	}

	size_t n, part, parts;
private:
	size_t index(size_t x) const {
		return n / parts * x + std::min(n % parts, x);
	}
};

struct KFold {
	typedef KFoldIterator iterator;
	KFold(int len, int n_folds): len(len), n_folds(n_folds) {}
	iterator begin() const {
		return KFoldIterator(len, 0, n_folds);
	}

	iterator end() const {
		return KFoldIterator(len, n_folds, n_folds);
	}

private:
	int len;
	int n_folds;
};

} // namespace faml
#endif //KFOLD_HPP
