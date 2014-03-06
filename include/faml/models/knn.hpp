#ifndef KNN_HPP
#define KNN_HPP

#include <unordered_map>
#include <stdexcept>
#include <algorithm>

#include "faml/data.hpp"

namespace faml {

template<typename DataType, typename LabelType>
class KNNClassifier {
public:
	KNNClassifier(size_t K, const DistanceFunction &dist, const KernelFunction &kernel): K(K), dist(dist), kernel(kernel) {
		if (K == 0) {
			throw std::invalid_argument("K must be non-zero");
		}
	}

	void train(const std::vector<DataType> &samples, const std::vector<LabelType> &labels): samples(samples), labels(labels) {
	}

	LabelType predict(const DataType &sample) {
		if (samples.empty()) {
			throw std::logic_error("Classifier is not trained");
		}

		std::priority_queue< std::pair<double, size_t> > nearestNeighbors;
		for (size_t i = 0; i < dataset.size(); ++i) {
			double distance = dist(sample, samples[i]);
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
			labelWeight[labels[neighbor.second]] += kernel(neighbor.first);
		}

		return std::max_element(labelWeight.begin(), labelWeight.end(), labelWeightPairComparator)->first;
	}

	std::vector<LabelType> predict(const std::vector<DataType> &data) {
		std::vector<LabelType> labels;
		for (auto &sample : data) {
			labels.push_back(predict(sample));
		}
		return labels;
	}

private:
	static bool labelWeightPairComparator(const std::pair<LabelType, double> &lhs, const std::pair<LabelType, double> &rhs) {
		return lhs.second < rhs.second;
	}

	size_t K;
	const DistanceFunction &dist;
	const KernelFunction &kernel;
	std::vector<DataType> samples;
	std::vector<LabelType> labels;
};

} // namespace faml

#endif // KNN_HPP
