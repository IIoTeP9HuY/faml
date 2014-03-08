#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <eigen3/Eigen/Core>

#include "faml/data.hpp"

namespace faml {

Eigen::MatrixXf covarianceMatrix(const Table<Eigen::VectorXf> &samples) {
	size_t featuresDimension = samples.columnsNumber();

	Eigen::MatrixXf samplesMatrix(samples.rowsNumber(), featuresDimension);
	for (size_t i = 0; i < samples.rowsNumber(); ++i) {
		samplesMatrix.row(i) = samples[i];
	}

	Eigen::MatrixXf centered = samplesMatrix.rowwise() - samplesMatrix.colwise().mean();
	return (centered.adjoint() * centered) / double(samplesMatrix.rows());
}

} // namespace faml

#endif // STATISTICS_HPP
