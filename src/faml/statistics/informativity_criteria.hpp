#ifndef INFORMATIVITY_CRITERIA_HPP
#define INFORMATIVITY_CRITERIA_HPP

#include <unordered_map>
#include <cmath>
#include <iostream>

#include "variation_indices.hpp"

namespace faml {

template<typename Label>
struct InformativityCriteria {
	virtual ~InformativityCriteria() {}

	virtual double operator () (const TableView<Label> &positive, const TableView<Label> &negative) = 0;

};

template<typename Label>
struct EntropyCriteria : public InformativityCriteria<Label> {
	~EntropyCriteria() {}

	inline double h(double x) const {
		if (x == 0) {
			return 0;
		}
		return -x * log2(x);
	}

	double operator () (const TableView<Label> &positive, const TableView<Label> &negative) {
		std::unordered_set<Label> labels;

		std::unordered_map<Label, size_t> positiveLabelCount;
		for (const auto &label : positive) {
			++positiveLabelCount[label];
			labels.insert(label);
		}

		std::unordered_map<Label, size_t> negativeLabelCount;
		for (const auto &label : negative) {
			++negativeLabelCount[label];
			labels.insert(label);
		}

		size_t samplesNumber = positive.rowsNumber() + negative.rowsNumber();
		double entropy = 0;
		double positivePart = positive.rowsNumber() * 1.0 / samplesNumber;
		double negativePart = negative.rowsNumber() * 1.0 / samplesNumber;

		for (const auto &label : labels) {
			size_t positiveClassSize = positiveLabelCount[label];
			size_t negativeClassSize = negativeLabelCount[label];
			size_t classSize = positiveClassSize + negativeClassSize;

			entropy += h(classSize * 1.0 / samplesNumber);
			if (positive.rowsNumber()) {
				entropy -= positivePart * h(positiveClassSize * 1.0 / positive.rowsNumber());
			}
			if (negative.rowsNumber()) {
				entropy -= negativePart * h(negativeClassSize * 1.0 / negative.rowsNumber());
			}
		}
		return entropy;
	}
};

} // namespace faml

#endif // INFORMATIVITY_CRITERIA_HPP
