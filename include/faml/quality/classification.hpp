#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

#include <stdexcept>
#include <set>

#include "faml/data/table.hpp"

namespace faml {

template<typename LabelType>
size_t countCorrectPredictions(const TableView<LabelType> &realLabels, const TableView<LabelType> &predictedLabels) {
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
double accuracyScore(const TableView<LabelType> &realLabels, const TableView<LabelType> &predictedLabels) {
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
ConfusionMatrix<LabelType> confusionMatrix(const TableView<LabelType> &realLabels,
											const TableView<LabelType> &predictedLabels) {
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

template<typename LabelType>
class MeanScorer {
	typedef std::function<double(const TableView<LabelType> &realLabels,
								 const TableView<LabelType> &predictedLabels)> ScoreFunction;
public:
	MeanScorer(const ScoreFunction &scoreFunction):
		scoreFunction(scoreFunction), totalScore(0), updatesNumber(0) {}

	double score() {
		if (updatesNumber == 0) {
			throw std::logic_error("Can't calculate score without samples");
		}

		return totalScore / updatesNumber;
	}

	void updateScore(const TableView<LabelType> &realLabels, const TableView<LabelType> &predictedLabels) {
		totalScore += scoreFunction(realLabels, predictedLabels);
		++updatesNumber;
	}

private:
	ScoreFunction scoreFunction;
	double totalScore;
	size_t updatesNumber;
};

template<typename LabelType>
class AccuracyScorer : public MeanScorer<LabelType> {
public:
	AccuracyScorer(): MeanScorer<LabelType>(accuracyScore<LabelType>) {}
};

} // namespace faml

#endif // CLASSIFICATION_HPP
