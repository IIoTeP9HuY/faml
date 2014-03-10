#ifndef CROSS_VALIDATION_HPP
#define CROSS_VALIDATION_HPP
#include "faml/models/predictor.hpp"

namespace faml {

template <typename Row, typename Label, typename CVStrategy, typename Scorer>
auto crossValidate(const Predictor<Row, Label>& predictor, const TableView<Row>& table, const TableView<Label> labels, CVStrategy strategy, Scorer scorer) -> decltype(scorer.score()) {
	for(const auto& indices: strategy) {
		predictor.fit(table[indices.first], labels[indices.first]);
		auto prediction = predictor.predict(table[indices.second]);
		scorer.update(labels[indices.second], prediction);
	}
	return scorer.score();
}

} //namespace faml
#endif //CROSS_VALIDATION_HPP
