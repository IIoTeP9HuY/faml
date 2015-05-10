#pragma once

#include <string>
#include <fstream>
#include <memory>

namespace faml {

class FileReader {
public:
	FileReader(const std::string& filename): ifs(filename) {
		if (!ifs.is_open()) {
			throw std::ios_base::failure("File doesn't exist: " + filename);
		}
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

template<typename FileReader>
class ReaderFactory {
public:
	std::unique_ptr<FileReader> getInstance(const std::string& filename) const {
		return std::unique_ptr<FileReader>(new FileReader(filename));
	}
};


} // namespace faml
