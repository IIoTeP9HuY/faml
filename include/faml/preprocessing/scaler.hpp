#ifndef SCALER_HPP
#define SCALER_HPP

#include "faml/data.hpp"
#include "eigen3/Eigen/Core"

namespace faml {

template<typename DataType>
class Scaler {
public:
	Scaler() {}
	virtual ~Scaler() {}

	virtual void train(const Table<DataType> &samples) = 0;

	virtual DataType operator () (const DataType &sample) const = 0;

	virtual std::string toString() const = 0;
};

class NormalScaler : public Scaler<Eigen::VectorXf> {
	typedef Eigen::VectorXf DataType;
public:
	NormalScaler() {}
	~NormalScaler() {}

	void train(const Table<DataType> &samples) {
		size_t F = samples.columnsNumber();
		mean = DataType(F);
		deviation = DataType(F);

		for (const auto &sample : samples) {
			for (size_t i = 0; i < F; ++i) {
				mean[i] += sample[i] * 1.0 / samples.rowsNumber();
			}
		}
		for (const auto &sample : samples) {
			for (size_t i = 0; i < F; ++i) {
				deviation[i] += (sample[i] - mean[i]) * (sample[i] - mean[i])
								 / samples.rowsNumber() / samples.rowsNumber();
			}
		}
		for (size_t i = 0; i < F; ++i) {
			deviation[i] = sqrt(deviation[i]);
		}
	}

	DataType operator () (const DataType &sample) const {
		return (sample - mean).cwiseQuotient(deviation);
	}

	std::string toString() const {
		return "NormalScaler";
	}

private:
	DataType mean;
	DataType deviation;
};

class MinMaxScaler : public Scaler<Eigen::VectorXf> {
	typedef Eigen::VectorXf DataType;
public:
	MinMaxScaler(const DataType &lowerBound, const DataType &upperBound):
		lowerBound(lowerBound), upperBound(upperBound) {}

	MinMaxScaler(size_t n, double lowerBound, double upperBound): lowerBound(VectorXf::Ones(n) * lowerBound),
	                                                              upperBound(VectorXf::Ones(n) * upperBound){
	}
	virtual ~MinMaxScaler() {}

	void train(const Table<DataType> &samples) {
		if (!samples.empty()) {
			minValues = samples[0];
			maxValues = samples[0];
		}

		for (const auto &sample : samples) {
			minValues = minValues.cwiseMin(sample);
			maxValues = maxValues.cwiseMax(sample);
		}
	}

	DataType operator () (const DataType &sample) const {
		return (sample - minValues).cwiseQuotient(maxValues - minValues).cwiseProduct(upperBound - lowerBound) + lowerBound;
	}

	std::string toString() const {
		return "MinMaxScaler";
	}

private:
	DataType lowerBound, upperBound;
	DataType minValues, maxValues;
};

template <typename T>
class DummyScaler : public Scaler<T> {
public:
	virtual ~DummyScaler() {}

	virtual void train(const Table<T>&) {}
	virtual T operator () (const T& sample) const { return sample; }
	virtual std::string toString() const {
		return "DummyScaler";
	}

};

} // namespace faml

#endif // SCALER_HPP
