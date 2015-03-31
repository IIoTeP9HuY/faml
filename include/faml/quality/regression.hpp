#pragma once

#include <cmath>

namespace faml {

double rmseLoss(const TableView<double>& realLabels, const TableView<double>& predictedLabels) {
	if (realLabels.rowsNumber() != predictedLabels.rowsNumber()) {
		throw std::invalid_argument("Arguments should have same length");
	}

	double loss = 0.0;
	for (size_t i = 0; i < realLabels.rowsNumber(); ++i) {
		double delta = realLabels[i] - predictedLabels[i];
		loss += delta * delta;
	}
	return sqrt(loss / realLabels.rowsNumber());
}

} // namespace faml
