#pragma once

#include "faml/models/predictor.hpp"

#include <eigen3/Eigen/Core>

#include <iostream>

namespace faml {

using Eigen::VectorXf;

class LinearRegressor : public Predictor<VectorXf, double> {
public:
	LinearRegressor(int iterationNumber): iterationNumber(iterationNumber) {}

	void train(const TableView<VectorXf>& samples, const TableView<double>& values) override {
		featuresNumber = samples.columnsNumber();
		weights = VectorXf::Zero(featuresNumber);

		double learning_rate = base_learning_rate;
		for (int iteration = 0; iteration < iterationNumber; ++iteration) {
			auto sampleIt = samples.begin();
			auto valueIt = values.begin();

			while (sampleIt != samples.end() && valueIt != values.end()) {
				const VectorXf& sample = *sampleIt;
				double value = *valueIt;
				weights -= learning_rate * 2 * (weights.dot(sample) - value) * sample;
				++sampleIt;
				++valueIt;
			}
			learning_rate *= decay;
		}
	}

	double predict(const VectorXf& sample) override {
		return weights.dot(sample);
	}

	using Predictor::predict;

private:
	int featuresNumber;
	VectorXf weights;

	int iterationNumber;
	double base_learning_rate = 0.3;
	double decay = 0.99;
};

} // namespace faml
