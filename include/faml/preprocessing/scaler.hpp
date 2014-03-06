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
			// FIX THIS
			deviation[i] = 1.0 / sqrt(deviation[i]);
		}
	}

	DataType operator () (const DataType &sample) const {
		return (sample - mean).cwiseQuotient(deviation);
	}

private:
	DataType mean;
	DataType deviation;
};

} // namespace faml

#endif // SCALER_HPP
