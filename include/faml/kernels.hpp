#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <cmath>

namespace faml {

const double eps = 1e-9;

struct KernelFunction {
	virtual ~KernelFunction() {}
	virtual double operator ()(const double dist) const = 0;
};

struct InverseKernel : public KernelFunction {
	InverseKernel(double lambda): lambda(lambda) {}

	double operator ()(const double dist) const {
		return lambda / dist;
	}

	double lambda;
};

struct DiscreteKernel : public KernelFunction {
	DiscreteKernel() {}

	double operator ()(const double dist) const {
		return (dist <= 1 + eps);
	}
};

struct RBFKernel : public KernelFunction {
	RBFKernel(double beta): beta(beta) {}

	double operator ()(const double dist) const {
		return exp(-beta * dist * dist);
	}

	double beta;
};

struct TriangleKernel : public KernelFunction {
	TriangleKernel() {}

	double operator ()(const double dist) const {
		return (1 - dist) * (dist <= 1 + eps);
	}
};

struct QuarticKernel : public KernelFunction {
	QuarticKernel() {}

	double operator ()(const double dist) const {
		return (15.0 / 16.0) * (1 - dist * dist) * (1 - dist  * dist) * (dist <= 1 + eps);
	}
};

struct EpanechnikovKernel : public KernelFunction {
	EpanechnikovKernel() {}

	double operator ()(const double dist) const {
		return (0.75) * (1 - dist * dist) * (dist <= 1 + eps);
	}
};

} // namespace faml

#endif // KERNELS_HPP
