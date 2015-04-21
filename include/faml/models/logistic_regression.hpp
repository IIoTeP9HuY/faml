#pragma once

#include "faml/models/predictor.hpp"

#include <eigen3/Eigen/Core>

#include <iostream>

namespace faml {

template<typename DataType>
class LogisticRegressor : public Predictor<DataType, double> {
public:
	LogisticRegressor(int featuresNumber, int iterationNumber):
		featuresNumber(featuresNumber),
		iterationNumber(iterationNumber)
	{}

	void train(const TableView<DataType>& samples, const TableView<double>& labels) override {
		weights = DataType(featuresNumber);

		double learning_rate = base_learning_rate;
		int samplesProcessed = 0;
		int pow2 = 1;
		for (int iteration = 0; iteration < iterationNumber; ++iteration) {
			auto sampleIt = samples.begin();
			auto labelIt = labels.begin();

			while (sampleIt != samples.end() && labelIt != labels.end()) {
				const DataType& sample = *sampleIt;
				double label = *labelIt;
				double prediction = 1 / (1 + exp(-weights.dot(sample)));
				weights -= learning_rate * 2 * (label - prediction) * sample;
				++sampleIt;
				++labelIt;
				++samplesProcessed;
				if (samplesProcessed >= pow2) {
					std::cerr << "Samples processed: " << samplesProcessed << std::endl;
					pow2 *= 2;
				}
			}

			auto predictedLabels = predict(samples);
			double loss = logLossScore(labels, predictedLabels);
			std::cerr << "Iteration: " << iteration << ", loss: " << loss << std::endl;

			learning_rate *= decay;
		}
	}

	double predict(const DataType& sample) override {
		return 1 / (1 + exp(weights.dot(sample)));
	}

	using Predictor<DataType, double>::predict;

private:
	int featuresNumber;
	DataType weights;

	int iterationNumber;
	double base_learning_rate = 0.3;
	double decay = 0.99;
};

} // namespace faml
