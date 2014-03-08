#ifndef DISTANCES_HPP
#define DISTANCES_HPP

#include <eigen3/Eigen/Core>
#include <stdexcept>

namespace faml {

using namespace Eigen;

inline double fastpow(double x, int power) {
	double result = 1.0;
	for (int i = 0; i < power; ++i) {
		result *= x;
	}	
	return result;
}

template<typename T>
struct DistanceFunction {
	virtual double operator ()(const T &lhs, const T &rhs) const {}
};

template<typename T>
struct MinkowskiDistance : DistanceFunction<T> {
	MinkowskiDistance(size_t power): power(power) {}

	double operator ()(const T &lhs, const T &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double distance = (rhs - lhs).cwiseAbs().array().pow(power).sum();
		return pow(distance, 1.0 / power);
	}

	size_t power;
};

template<typename T>
struct CosineDistance : DistanceFunction<T> {
	CosineDistance() {}

	double operator ()(const T &lhs, const T &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double scalarProduct = lhs.dot(rhs);
		double lhsNorm = lhs.norm();
		double rhsNorm = rhs.norm();

		double distance = 1 - scalarProduct / (lhsNorm * rhsNorm);

		return distance;
	}
};

template<typename T>
struct OverlapDistance : DistanceFunction<T> {
	OverlapDistance() {}

	double operator ()(const T &lhs, const T &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double mutualIntensity = lhs.cwiseMin(rhs).sum();
		double lhsNorm = lhs.sum();
		double rhsNorm = rhs.sum();

		double distance = 1 - 2 * mutualIntensity / (lhsNorm + rhsNorm);

		return distance;
	}
};

template<typename T>
struct WMinkowskiDistance : DistanceFunction<T> {
	WMinkowskiDistance(int power, const T &weights): power(power), weights(weights) {}

	double operator ()(const T &lhs, const T &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double distance = (weights.array() * (rhs - lhs).cwiseAbs().array().pow(power)).sum();

//		double distance = 0;

//		for (size_t i = 0; i < N; ++i) {
//			distance += weights(i) * fastpow(fabs(rhs[i] - lhs[i]), power);
//		}

		return pow(distance, 1.0 / power);
	}

	int power;
	T weights;
};

struct MahalanobisDistance : DistanceFunction<VectorXf> {
	MahalanobisDistance(const MatrixXf &inverseCovarianceMatrix):
		inverseCovarianceMatrix(inverseCovarianceMatrix) {}

	double operator ()(const VectorXf &lhs, const VectorXf &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double distance = 0;
		distance = sqrt((lhs - rhs).transpose() * inverseCovarianceMatrix * (lhs - rhs));
		return distance;
	}

	MatrixXf inverseCovarianceMatrix;
};

} // namespace faml

#endif // DISTANCES_HPP
