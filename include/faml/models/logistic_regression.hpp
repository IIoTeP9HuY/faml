#pragma once

#include "faml/models/predictor.hpp"
#include "faml/data/streaming_table.hpp"
#include "faml/quality/classification.hpp"

#include <eigen3/Eigen/Core>

#include <iostream>

namespace faml {

template<typename DataType, typename LabelType=double>
class LogisticRegressor /* : public Predictor<DataType, LabelType> */ {
	//using WeightsType = Eigen::VectorXf;
	using WeightsType = DataType;

public:
	LogisticRegressor(int featuresNumber, int iterationNumber):
		featuresNumber(featuresNumber),
		iterationNumber(iterationNumber)
	{}

	void train(const StreamingTable<std::pair<DataType, LabelType>>& samplesAndLabels) /* override */ {
		weights = WeightsType(featuresNumber);

		double learning_rate = base_learning_rate;
		for (int iteration = 0; iteration < iterationNumber; ++iteration) {
			auto it = samplesAndLabels.begin();

			double loss = 0;
			int samplesNumber = 0;

			while (it != samplesAndLabels.end()) {
				const DataType& sample = (*it).first;
				LabelType label = (*it).second;
				double prediction = predict(sample);
				loss += logLossValue(label, prediction);
				weights += sample * learning_rate * (label - prediction);
				++samplesNumber;
				++it;
			}
			loss /= samplesNumber;
			std::cerr << "Iteration: " << iteration << ", loss: " << loss << std::endl;

			learning_rate *= decay;
		}
	}

	double predict(const DataType& sample) /* override */ {
		return 1 / (1 + exp(-sample.dot(weights)));
	}

	//using Predictor<DataType, double>::predict;

private:
	int featuresNumber;
	WeightsType weights;

	int iterationNumber;
	double base_learning_rate = 0.3;
	double decay = 0.99;
};

} // namespace faml
