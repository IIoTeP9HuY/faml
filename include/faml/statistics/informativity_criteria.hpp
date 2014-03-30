#ifndef INFORMATIVITY_CRITERIA_HPP
#define INFORMATIVITY_CRITERIA_HPP

#include <unordered_map>
#include <cmath>

#include "variation_indices.hpp"

namespace faml {

template<typename Label>
struct InformationCriteria {
	InformationCriteria(const std::vector<Label> &labels): labels(labels) {}
	virtual ~VariationIndex() {}

	virtual double operator () (const TableView<Label> &positive, const TableView<Label> &negative) = 0;

	std::vector<Label> labels;
};

template<typename Label>
struct EntoropyCriteria : public InformationCriteria<Label> {
	EntoropyCriteria(const std::vector<Label> &labels): InformationCriteria<Label> (labels) {}
	~EntoropyCriteria() {}

	using InformationCriteria<Label>::labels;

	inline double h(double x) const {
		if (x == 0) {
			return 0;
		}
		return -x * log2(x);
	}

	double operator () (const TableView<Label> &positive, const TableView<Label> &negative) {
		std::unordered_map<Label, size_t> positiveLabelCount;
		for (const auto &label : positive) {
			++positiveLabelCount[label];
		}

		std::unordered_map<Label, size_t> negativeLabelCount;
		for (const auto &label : negative) {
			++negativeLabelCount[label];
		}

		size_t samplesNumber = positive.rowsNumber() + negative.rowsNumber();
		double entropy = 0;
		double positivePart = positive.rowsNumber() * 1.0 / samplesNumber;
		double negativePart = negative.rowsNumber() * 1.0 / samplesNumber;

		for (const auto &label : labels) {
			size_t positiveClassSize = positiveLabelCount[label];
			size_t negativeClassSize = negativeLabelCount[label];
			size_t classSize = positiveSize + negativeSize;

			entropy += h(classSize * 1.0 / samplesNumber);
			if (positive.rowsNumber()) {
				entropy -= positivePart * h(positiveClassSize / positive.rowsNumber());
			}
			if (negative.rowsNumber()) {
				entropy -= negativePart * h(negativeClassSize / negative.rowsNumber());
			}
		}
		return entropy;
	}
};

} // namespace faml

#endif // INFORMATIVITY_CRITERIA_HPP
