#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <sstream>
#include <algorithm>

#include "data/table.hpp"
#include "algebra/sparse_vector.hpp"

namespace faml {

std::vector<std::string> parseCSVHeader(const std::string& line) {
	std::vector<std::string> header;

	std::istringstream ss(line);
	std::string token;
	while (std::getline(ss, token, ',')) {
		header.push_back(token);
	}

	return header;
}

std::vector<std::string> readCSVHeader(std::ifstream &inputStream) {
	std::string line;
	std::getline(inputStream, line);
	return parseCSVHeader(line);
}

Table< std::vector<std::string> > readCSV(const std::string &filename,
											bool withHeader=true) {
	typedef std::vector<std::string> RowType;
	std::ifstream inputStream(filename);
	std::vector<std::string> header;
	if (withHeader) {
		header = readCSVHeader(inputStream);
	}
	size_t headerLength = header.size();
	Table<RowType> table(header);

	while (!inputStream.eof()) {
		std::string line;
		getline(inputStream, line);

		if (!headerLength) {
			headerLength = std::count(line.begin(), line.end(), ',') + 1;
			for (size_t i = 0; i < headerLength; ++i) {
				header.push_back(std::to_string(i));
			}
			table = Table<RowType>(header);
		}

		if (inputStream.eof()) {
			break;
		}

		std::istringstream ss(line);

		RowType data(headerLength);

		for (size_t i = 0; i < headerLength; ++i) {
			std::getline(ss, data[i], ',');
		}

		table.addRow(data);
	}
	return table;
}

class FileReader {
public:
	FileReader(const std::string& filename): ifs(filename) {
		// TODO(acid) Check if open
	}

	std::string getLine() {
		std::getline(ifs, buf);

		if (ifs.eof()) {
			return "";
		}

		return buf;
	}

private:
	std::string buf;
	std::ifstream ifs;
};

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

template<typename FileReader>
class ReaderFactory {
public:
	std::unique_ptr<FileReader> getInstance(const std::string& filename) const {
		return std::unique_ptr<FileReader>(new FileReader(filename));
	}
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

#endif // IO_HPP
