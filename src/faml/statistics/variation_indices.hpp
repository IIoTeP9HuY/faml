#ifndef VARIATION_INDICES_HPP
#define VARIATION_INDICES_HPP

#include <unordered_map>
#include <vector>

#include "faml/data/table.hpp"

namespace faml {

template<typename Label>
struct VariationIndex {
	VariationIndex(const std::vector<Label> &labels): labels(labels) {}
	virtual ~VariationIndex() {}

	virtual double operator () (const TableView<Label> &table) = 0;

	std::vector<Label> labels;
};

template<typename Label>
struct GiniIndex : public VariationIndex<Label> {
	GiniIndex(const std::vector<Label> &labels): VariationIndex<Label> (labels) {}
	~GiniIndex() {}

	using VariationIndex<Label>::labels;

	double operator () (const TableView<Label> &table) {
		std::unordered_map<Label, size_t> labelCount;

		for (const auto &label : table) {
			++labelCount[label];
		}

		double giniIndex = 0;
		for (size_t i = 0; i < labels.size(); ++i) {
			for (size_t j = i + 1; j < labels.size(); ++j) {
				giniIndex += abs(labelCount[labels[i]] - labelCount[labels[j]]);
			}
		}
		giniIndex = 1 - giniIndex / (labels.size() - 1) / table.rowsNumber();
		return giniIndex;
	}
};

} // namespace faml

#endif // VARIATION_INDICES_HPP
