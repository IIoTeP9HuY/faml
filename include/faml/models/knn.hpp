#ifndef KNN_HPP
#define KNN_HPP

#include <unordered_map>
#include <stdexcept>
#include <algorithm>

#include "faml/data.hpp"
#include "faml/distances.hpp"

namespace faml {

template<typename DataType, typename LabelType>
class KNNClassifier {
public:
	KNNClassifier(size_t K, const DistanceFunction<DataType> &dist, const KernelFunction &kernel): K(K), dist(dist), kernel(kernel) {
		if (K == 0) {
			throw std::invalid_argument("K must be non-zero");
		}
	}

	void train(const Table<DataType> &samples, const Table<LabelType> &labels) {
		this->baseSamples = samples;
		this->baseLabels = labels;
	}

	LabelType predict(const DataType &sample) {
		if (baseSamples.empty()) {
			throw std::logic_error("Classifier is not trained");
		}

		std::priority_queue< std::pair<double, size_t> > nearestNeighbors;
		for (size_t i = 0; i < baseSamples.rowsNumber(); ++i) {
			double distance = dist(sample, baseSamples[i]);
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
			labelWeight[baseLabels[neighbor.second]] += kernel(neighbor.first);
		}

		return std::max_element(labelWeight.begin(), labelWeight.end(), labelWeightPairComparator)->first;
	}

	Table<LabelType> predict(const Table<DataType> &samples) {
		Table<LabelType> labels(baseLabels.getColumnsNames());
		labels.resizeRows(samples.rowsNumber());

		#pragma omp parallel for
		for (size_t i = 0; i < samples.rowsNumber(); ++i) {
			labels[i] = predict(samples[i]);
		}
		return labels;
	}

private:
	static bool labelWeightPairComparator(const std::pair<LabelType, double> &lhs, const std::pair<LabelType, double> &rhs) {
		return lhs.second < rhs.second;
	}

	size_t K;
	const DistanceFunction<DataType> &dist;
	const KernelFunction &kernel;
	Table<DataType> baseSamples;
	Table<LabelType> baseLabels;
};

} // namespace faml

#endif // KNN_HPP
