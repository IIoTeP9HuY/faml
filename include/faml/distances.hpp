#ifndef DISTANCES_HPP
#define DISTANCES_HPP

#include <eigen3/Eigen/Core>
#include <stdexcept>
#include <string>

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
	virtual double operator ()(const T &lhs, const T &rhs) const = 0;
	virtual std::string toString() const = 0;
	virtual ~DistanceFunction(){}
};

struct EuclidianDistance : DistanceFunction<VectorXf> {
	double operator ()(const VectorXf &lhs, const VectorXf &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double distance = (rhs - lhs).norm();
		return distance;
	}

	std::string toString() const {
		return "EuclidianDistance";
	}

	virtual ~EuclidianDistance() {};
};

struct MinkowskiDistance : DistanceFunction<VectorXf> {
	MinkowskiDistance(size_t power): power(power) {}

	double operator ()(const VectorXf &lhs, const VectorXf &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double distance = 0;
		VectorXf delta = rhs - lhs;

		for (size_t i = 0; i < N; ++i) {
			distance += fastpow(fabs(delta[i]), power);
		}

		return pow(distance, 1.0 / power);
	}

	std::string toString() const {
		return "MinkowskiDistance, p = " + std::to_string(power);
	}

	virtual ~MinkowskiDistance() {}
private:
	size_t power;
};

struct CosineDistance : DistanceFunction<VectorXf> {
	CosineDistance() {}

	double operator ()(const VectorXf &lhs, const VectorXf &rhs) const {
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

	std::string toString() const {
		return "CosineDistance";
	}

	virtual ~CosineDistance() {}
};

struct OverlapDistance : DistanceFunction<VectorXf> {
	double operator ()(const VectorXf &lhs, const VectorXf &rhs) const {
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

	std::string toString() const {
		return "OverlapDistance";
	}
	
	virtual ~OverlapDistance() {}
};

struct WMinkowskiDistance : DistanceFunction<VectorXf> {
	WMinkowskiDistance(int power, const VectorXf &weights): power(power), weights(weights) {}

	double operator ()(const VectorXf &lhs, const VectorXf &rhs) const {
		size_t N = lhs.size();
		if (lhs.size() != rhs.size()) {
			throw std::invalid_argument("Arguments should have same length");
		}

		double distance = 0;
		VectorXf delta = rhs - lhs;

		for (size_t i = 0; i < N; ++i) {
			distance += weights(i) * fastpow(fabs(delta[i]), power);
		}

		return pow(distance, 1.0 / power);
	}

	std::string toString() const {
		return "WMinkowskiDistance, p = " + std::to_string(power);
	}
private:
	int power;
	VectorXf weights;
public:
	virtual ~WMinkowskiDistance() {}
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
	std::string toString() const {
		return "MahalanobisDistance";
	}

private:
	MatrixXf inverseCovarianceMatrix;
};

} // namespace faml

#endif // DISTANCES_HPP
