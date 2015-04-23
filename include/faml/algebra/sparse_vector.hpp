#pragma once

#include <unordered_map>

namespace faml {

template<typename T>
class SparseVector {
public:
	SparseVector(int maxSize)
	{ }

	SparseVector()
	{ }

	void setValue(int position, T value) {
		values[position] = value;
	}

	void addValue(int position, T value) {
		values[position] += value;
	}

	T getValue(int position) const {
		auto it = values.find(position);
		return it == values.end() ? 0 : it->second;
	}

	T dot(const SparseVector& rhs) const {
		T result = T();
		for (const auto& valuePair : values) {
			result += valuePair.second * rhs.getValue(valuePair.first);
		}
		return result;
	}

	SparseVector& operator += (const SparseVector& rhs) {
		for (const auto& valuePair : rhs.values) {
			addValue(valuePair.first, valuePair.second);
		}
		return *this;
	}

	SparseVector& operator *= (T rhs) {
		for (auto& valuePair : values) {
			valuePair.second *= rhs;
		}
		return *this;
	}

	SparseVector operator * (T rhs) const {
		SparseVector result = *this;
		result *= rhs;
		return result;
	}

private:
	std::unordered_map<int, T> values;
};

} // namespace faml
