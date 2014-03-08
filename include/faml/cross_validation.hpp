#ifndef CROSS_VALIDATION_HPP
#define CROSS_VALIDATION_HPP

namespace faml {

template<typename URNG>
std::pair<std::vector<size_t>, std::vector<size_t>> trainTestSplit(
		size_t n,
		size_t part,
		URNG&& gen
	) {
		typedef std::vector<size_t> V; 
		V indicies(n);
		iota(indicies.begin(), indicies.end(), 0);
		shuffle(indicies.begin(), indicies.end(), gen);
		return std::make_pair(V(indicies.begin(), indicies.begin() + part), V(indicies.begin() + part, indicies.end()));
	};

template<typename URNG>
std::pair<std::vector<size_t>, std::vector<size_t>> trainTestSplit(
		size_t n,
		double fraction,
		URNG&& gen
	) {
		return trainTestSplit(n, static_cast<size_t>(n * fraction), std::forward<URNG>(gen));
	};


} // namespace faml

#endif // DISTANCES_HPP
