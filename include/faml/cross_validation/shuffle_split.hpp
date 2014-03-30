#ifndef SHUFFLESPLIT_HPP
#define SHUFFLESPLIT_HPP

#include <stdexcept>
#include <stdint.h>

namespace faml {

struct ShuffleSplitIterator {
	ShuffleSplitIterator(size_t len, size_t train, size_t iterations, uint_fast32_t seed): len(len), train(train), iterations(iterations), seed(seed) {}
	
	bool operator == (const ShuffleSplitIterator& rhs) const {
		return len == rhs.len && train == rhs.train && iterations == rhs.iterations;
	}
	bool operator != (const ShuffleSplitIterator& rhs) const {
		return !(*this == rhs);
	}

	ShuffleSplitIterator& operator ++ () {
		seed = std::mt19937(seed)();
		--iterations;
		return *this;
	}

	std::pair<std::vector<size_t>, std::vector<size_t>> operator * () const {
		std::mt19937 gen(seed);
		std::vector<size_t> indicies(len);
		std::iota(indicies.begin(), indicies.end(), 0);
		std::shuffle(indicies.begin(), indicies.end(), gen);
		return std::make_pair(std::vector<size_t>(indicies.begin(), indicies.begin() + train), std::vector<size_t>(indicies.begin() + train, indicies.end()));
	}

private:
	int len, train, iterations;
	uint_fast32_t seed;
};

struct ShuffleSplit {
	typedef ShuffleSplitIterator iterator;
	ShuffleSplit(size_t len, double train, size_t iterations, uint_fast32_t seed = 0): ShuffleSplit(len, static_cast<size_t>(train * len), iterations, seed) {}
	ShuffleSplit(size_t len, size_t train, size_t iterations, uint_fast32_t seed = 0): len(len), train(train), iterations(iterations), seed(seed) {}

	iterator begin() const {
		return iterator(len, train, iterations, seed);
	}

	iterator end() const {
		return iterator(len, train, 0, seed);
	}

private:
	size_t len;
	size_t train;
	size_t iterations;
	uint_fast32_t seed;
};

std::pair<std::vector<size_t>, std::vector<size_t>> trainTestSplit(
		size_t n,
		size_t part,
		uint_fast32_t seed = 0
	) {
		return *ShuffleSplit(n, part, 1, seed).begin();
	};

std::pair<std::vector<size_t>, std::vector<size_t>> trainTestSplit(
		size_t n,
		double fraction,
		uint_fast32_t seed = 0
	) {
		return trainTestSplit(n, static_cast<size_t>(n * fraction), seed);
	};


} // namespace faml

#endif // SHUFFLESPLIT_HPP
