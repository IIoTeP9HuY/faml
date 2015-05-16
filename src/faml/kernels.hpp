#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cmath>
#include <string>

namespace faml {

const double eps = 1e-9;

struct KernelFunction {
	virtual ~KernelFunction() {}
	virtual double operator ()(const double dist) const = 0;
	virtual std::string toString() const = 0;
};

struct InverseKernel : public KernelFunction {
	double operator ()(const double dist) const {
		return 1.0 / dist;
	}

	std::string toString() const {
		return "InverseKernel";
	}
};

struct DiscreteKernel : public KernelFunction {
	double operator ()(const double dist) const {
		return (dist <= 1 + eps);
	}

	std::string toString() const {
		return "DiscreteKernel";
	}
};

struct RBFKernel : public KernelFunction {
	RBFKernel(double beta): beta(beta) {}

	double operator ()(const double dist) const {
		return exp(-beta * dist * dist);
	}

	std::string toString() const {
		return "RBFKernel, beta = " + std::to_string(beta);
	}

	double beta;
};

struct TriangleKernel : public KernelFunction {
	TriangleKernel() {}

	double operator ()(const double dist) const {
		return (1 - dist) * (dist <= 1 + eps);
	}

	std::string toString() const {
		return "TriangularKernel";
	}
};

struct QuarticKernel : public KernelFunction {
	QuarticKernel() {}

	double operator ()(const double dist) const {
		return (15.0 / 16.0) * (1 - dist * dist) * (1 - dist  * dist) * (dist <= 1 + eps);
	}

	std::string toString() const {
		return "QuarticKernel";
	}
};

struct EpanechnikovKernel : public KernelFunction {
	EpanechnikovKernel() {}

	double operator ()(const double dist) const {
		return (0.75) * (1 - dist * dist) * (dist <= 1 + eps);
	}

	std::string toString() const {
		return "EpanechnikovKernel";
	}
};

} // namespace faml

#endif // KERNELS_HPP
