#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <stdexcept>
#include <set>

#include "faml/data.hpp"

namespace faml {

template<typename LabelType>
size_t countCorrectPredictions(const Table<LabelType> &realLabels, const Table<LabelType> &predictedLabels) {
	if (realLabels.rowsNumber() != predictedLabels.rowsNumber()) {
		throw std::invalid_argument("Arguments should have same length");
	}

	size_t correctPredictionsNumber = 0;
	for (size_t i = 0; i < realLabels.rowsNumber(); ++i) {
		correctPredictionsNumber += (realLabels[i] == predictedLabels[i]);
	}
	return correctPredictionsNumber;
}

template<typename LabelType>
double accuracyScore(const Table<LabelType> &realLabels, const Table<LabelType> &predictedLabels) {
	size_t correctPredictionsNumber = countCorrectPredictions(realLabels, predictedLabels);

	if (realLabels.rowsNumber() == 0) {
		return 1.0;
	}
	return correctPredictionsNumber * 1.0 / realLabels.rowsNumber();
}

template<typename LabelType>
class ConfusionMatrix {
public:
	std::unordered_map< LabelType, std::unordered_map<LabelType, size_t> > matrix;

	ConfusionMatrix(const std::vector<LabelType> &labels): labels(labels) {
	}

	void addSample(const LabelType &real, const LabelType &predicted) {
		++matrix[real][predicted];
	}

	template<typename T>
	friend std::ostream operator << (std::ostream &os, const ConfusionMatrix<T> &);
private:
	std::vector<LabelType> labels;
};

template<typename LabelType>
std::ostream operator << (std::ostream &os, const ConfusionMatrix<LabelType> &confusionMatrix) {
	os << "  ";
	for (const LabelType &label : confusionMatrix.labels) {
		os << std::setw(4) << label << " ";
	}
	os << std::endl;

	for (const LabelType &firstLabel : confusionMatrix.labels) {
		os << firstLabel << " ";
		for (const LabelType &secondLabel : confusionMatrix.labels) {
			os << std::setw(4) << confusionMatrix.matrix[firstLabel][secondLabel] << " ";
		}
		os << std::endl;
	}
}

template<typename LabelType>
ConfusionMatrix<LabelType> confusionMatrix(const Table<LabelType> &realLabels,
											const Table<LabelType> &predictedLabels) {
	if (realLabels.rowsNumber() != predictedLabels.rowsNumber()) {
		throw std::invalid_argument("Arguments should have same length");
	}
	std::set<LabelType> labels;

	for (const LabelType &label : realLabels) {
		labels.insert(label);
	}
	for (const LabelType &label : predictedLabels) {
		labels.insert(label);
	}

	ConfusionMatrix<LabelType> result(std::vector<LabelType>(labels.begin(), labels.end()));
	for (size_t i = 0; i < realLabels.rowsNumber(); ++i) {
		result.addSample(realLabels[i], predictedLabels[i]);
	}
	return result;
}

} // namespace faml

#endif // CLASSIFICATION_HPP
