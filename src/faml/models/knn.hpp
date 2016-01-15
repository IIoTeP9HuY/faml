#ifndef KNN_HPP
#define KNN_HPP

#include <unordered_map>
#include <stdexcept>
#include <algorithm>

#include "faml/data/table.hpp"
#include "faml/distances.hpp"
#include "faml/models/predictor.hpp"

namespace faml {

template<typename DataType, typename LabelType>
class KNNClassifier : public Predictor<DataType, LabelType> {
public:
	KNNClassifier(size_t K, std::shared_ptr< DistanceFunction<DataType> > dist, std::shared_ptr<KernelFunction> kernel):
		K(K), dist(std::move(dist)), kernel(std::move(kernel)) {
		if (K == 0) {
			throw std::invalid_argument("K must be non-zero");
		}
	}

	void train(const TableView<DataType> &samples, const TableView<LabelType> &labels) override {
		this->baseSamples = samples.toTable();
		this->baseLabels = labels.toTable();
	}

	LabelType predict(const DataType &sample) override {
		if (baseSamples.empty()) {
			throw std::logic_error("Classifier is not trained");
		}

		std::priority_queue< std::pair<double, size_t> > nearestNeighbors;
		for (size_t i = 0; i < baseSamples.rowsNumber(); ++i) {
			double distance = (*dist)(sample, baseSamples[i]);
			nearestNeighbors.push(std::make_pair(distance, i));
			if (nearestNeighbors.size() > K) {
				nearestNeighbors.pop();
			}
		}

		std::unordered_map<LabelType, double> labelWeight;

		double maxDistance = nearestNeighbors.top().first;
		while (!nearestNeighbors.empty()) {
			std::pair<double, size_t> neighbor = nearestNeighbors.top();
			nearestNeighbors.pop();
			neighbor.first /= maxDistance;
			labelWeight[baseLabels[neighbor.second]] += (*kernel)(neighbor.first);
		}

		return std::max_element(labelWeight.begin(), labelWeight.end(), labelWeightPairComparator)->first;
	}

	Table<LabelType> predict(const TableView<DataType> &samples) override {
		std::vector<LabelType> predictions(samples.rowsNumber());

		#pragma omp parallel for
		for (size_t i = 0; i < samples.rowsNumber(); ++i) {
			predictions[i] = predict(samples[i]);
		}

		return Table<LabelType> (baseLabels.columnsNames(),
								 std::make_move_iterator(predictions.begin()),
								 std::make_move_iterator(predictions.end()));
	}

	virtual ~KNNClassifier() {}
private:
	static bool labelWeightPairComparator(const std::pair<LabelType, double> &lhs, const std::pair<LabelType, double> &rhs) {
		return lhs.second < rhs.second;
	}

	size_t K;
	std::shared_ptr< DistanceFunction<DataType> > dist;
	std::shared_ptr<KernelFunction> kernel;
	Table<DataType> baseSamples;
	Table<LabelType> baseLabels;
};

} // namespace faml

#endif // KNN_HPP
