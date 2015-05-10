#pragma once

#include "io.hpp"
#include "readers.hpp"

#include "faml/algebra/sparse_vector.hpp"

#include <vector>

namespace faml {

class CSVToSparseParser {
public:
	CSVToSparseParser(
			std::unique_ptr<FileReader> fileReader,
			const std::string& labelColumn,
			int hashSpaceSize)
		: fileReader(std::move(fileReader))
		, labelColumn(labelColumn)
		, hashSpaceSize(hashSpaceSize)
	{
		readHeader();
	}

	void readHeader() {
		header = parseCSVHeader(fileReader->getLine());
		for (size_t i = 0; i < header.size(); ++i) {
			if (header[i] == labelColumn) {
				labelColumnIndex = i;
			}
		}
		if (labelColumnIndex == -1) {
			throw std::logic_error("Label column not found");
		}
	}

	bool isFinished() const {
		return finished;
	}

	int getRowNumber() const {
		return rowNumber;
	}

	std::pair<SparseVector<float>, float> getRow() {
		std::string line;
		if (!finished) {
			line = fileReader->getLine();
		}
		if (line.empty()) {
			finished = true;
			return std::pair<SparseVector<float>, float>();
		}
		++rowNumber;
		return parseLine(line);
	}

private:
	std::pair<SparseVector<float>, float> parseLine(const std::string& line) const {
		auto hasher = std::hash<std::string>();
		auto row = std::make_pair(SparseVector<float>(hashSpaceSize), 0);
		
		std::istringstream ss(line);
		std::string string_feature;
		for (int i = 0; i < static_cast<int>(header.size()); ++i) {
			std::getline(ss, string_feature, ',');
			if (i != labelColumnIndex) {
				string_feature = header[i] + "_" + string_feature;
				int feature = hasher(string_feature) % hashSpaceSize;
				row.first.setValue(feature, 1);
			} else {
				row.second = std::stod(string_feature);
			}
		}
		return std::move(row);
	}

	std::unique_ptr<FileReader> fileReader;
	std::string labelColumn;
	int hashSpaceSize;
	int labelColumnIndex = -1;
	std::vector<std::string> header;
	bool finished = false;
	int rowNumber = 0;
};


class CSVToSparseParserFactory {
public:
	typedef CSVToSparseParser Parser;

	CSVToSparseParserFactory(const std::string& labelColumn, int hashSpaceSize)
		: labelColumn(labelColumn)
		, hashSpaceSize(hashSpaceSize)
	{ }

	std::unique_ptr<CSVToSparseParser> getInstance(std::unique_ptr<FileReader> fileReader) const {
		return std::unique_ptr<CSVToSparseParser>(
			new CSVToSparseParser(std::move(fileReader), labelColumn, hashSpaceSize));
	}

private:
	std::string labelColumn;
	int hashSpaceSize;
};

} // namespace faml
