#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <vector>
#include <unordered_set>
#include <cmath>

using namespace std;

typedef std::vector< std::vector<size_t> > Image;

const size_t H = 28;
const size_t W = 28;

bool isTraining;
size_t samplesNumber = 20000; 

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt(double((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

const int dh[4] = {1, 0, -1, 0};
const int dw[4] = {0, 1, 0, -1};

struct TrainObject
{
	TrainObject() {}
	TrainObject(const Image& image, size_t classId = -1): image(image), 
	                                                      classId(classId) {}


	std::vector<double> buildFeatureVector()
	{
		calculateBoundingBox();
		calculateMassCenter();
		std::vector<double> featureVector;

		// for (int h = 0; h < H; ++h)
		// {
		// 	for (int w = 0; w < W; ++w)
		// 	{
		// 		// featureVector.push_back(isPainted(h, w));
		// 		featureVector.push_back(image[h][w]);
		// 	}
		// }

		int partHNumber = 14;
		int partWNumber = 14;
		int partHeight = (H + partHNumber - 1) / partHNumber;
		int partWidth = (W + partWNumber - 1) / partWNumber;

		for (int i = 0; i < partHNumber; ++i)
		{
			for (int j = 0; j < partWNumber; ++j)
			{
				int hMin = min(0 + i * partHeight, (int) H);
				int hMax = min(0 + (i + 1) * partHeight, (int) H);
				int wMin = min(0 + j * partWidth, (int) W);
				int wMax = min(0 + (j + 1) * partWidth, (int) W);

				featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
			}
		}

		partHNumber = 1;
		partWNumber = 7;
		partHeight = (H + partHNumber - 1) / partHNumber;
		partWidth = (W + partWNumber - 1) / partWNumber;

		for (int i = 0; i < partHNumber; ++i)
		{
			for (int j = 0; j < partWNumber; ++j)
			{
				int hMin = min(0 + i * partHeight, (int) H);
				int hMax = min(0 + (i + 1) * partHeight, (int) H);
				int wMin = min(0 + j * partWidth, (int) W);
				int wMax = min(0 + (j + 1) * partWidth, (int) W);

				featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
			}
		}

		partHNumber = 7;
		partWNumber = 1;
		partHeight = (H + partHNumber - 1) / partHNumber;
		partWidth = (W + partWNumber - 1) / partWNumber;

		for (int i = 0; i < partHNumber; ++i)
		{
			for (int j = 0; j < partWNumber; ++j)
			{
				int hMin = min(0 + i * partHeight, (int) H);
				int hMax = min(0 + (i + 1) * partHeight, (int) H);
				int wMin = min(0 + j * partWidth, (int) W);
				int wMax = min(0 + (j + 1) * partWidth, (int) W);

				featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
			}
		}

		partHNumber = 4;
		partWNumber = 4;
		partHeight = (H + partHNumber - 1) / partHNumber;
		partWidth = (W + partWNumber - 1) / partWNumber;

		for (int i = 0; i < partHNumber; ++i)
		{
			for (int j = 0; j < partWNumber; ++j)
			{
				int hMin = min(0 + i * partHeight, (int) H);
				int hMax = min(0 + (i + 1) * partHeight, (int) H);
				int wMin = min(0 + j * partWidth, (int) W);
				int wMax = min(0 + (j + 1) * partWidth, (int) W);

				featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
			}
		}

		// for (int h = H / 3; h < H - H / 3; ++h)
		// {
		// 	for (int w = W / 3; w < W - W / 3; ++w)
		// 	{
		// 		featureVector.push_back(isPainted(h, w));
		// 	}
		// }

		// for (int h = 13; h < 17; ++h)
		// {
		// 	featureVector.push_back(calcMeanInRow(h));
		// }
		// for (int w = 10; w < 20; ++w)
		// {
		// 	featureVector.push_back(calcMeanInColumn(w));
		// }
		// for (int h = 10; h < 15; ++h)
		// {
		// 	featureVector.push_back(calcMaxInRow(h) - calcMinInRow(h));
		// }
		// for (int w = 6; w < 8; ++w)
		// {
		// 	featureVector.push_back(calcMaxInColumn(w) - calcMaxInColumn(w));
		// }

		size_t imageH = hMaxBound - hMinBound;
		size_t imageW = wMaxBound - wMinBound;
		double imageSidesRatio = imageH / imageW;

		double meanDistanceToMassCenter = calculateMeanDistanceToMassCenter();
		double meanDistanceToMassCenterPainted = calculateMeanDistanceToMassCenterPainted();

		featureVector.push_back(hMassCenter - hMinBound);
		featureVector.push_back(wMassCenter - wMinBound);

		//std::cerr << "class: " << classId << ", meanDistance: " << meanDistanceToMassCenter / (imageH + imageW) << std::endl;
		// std::cerr << "class: " << classId << std::endl;
		featureVector.push_back(meanDistanceToMassCenter / (imageH + imageW));

		for (size_t R = 1; R < 10; R += 2)
		{
			// std:cerr << R << ": " << calculatePointsNumberInCircle(hMassCenter, wMassCenter, R) << std::endl;
			featureVector.push_back(calculatePointsNumberInCircle(hMassCenter, wMassCenter, R));
		}
		// std::cerr << "-----------------" << std::endl;
		//std::cerr << "class: " << classId << ", meanDistance: " << "(" << meanDistanceToMassCenter << ", " << meanDistanceToMassCenterPainted << ")" << std::endl;
		featureVector.push_back(meanDistanceToMassCenter);
		// featureVector.push_back(meanDistanceToMassCenterPainted);

		featureVector.push_back(imageH);
		featureVector.push_back(imageW);
		// featureVector.push_back(imageSidesRatio);

		size_t perimeter = calculatePerimeter();

		featureVector.push_back(perimeter);

		double meanPixelChangesNumberInColumns = calculateMeanPixelChangesNumberInColumns(wMinBound, wMaxBound);
		double meanPixelChangesNumberInRows = calculateMeanPixelChangesNumberInRows(hMinBound, hMaxBound);

		size_t maxPointsNumberInColumns;
		double maxPixelChangesNumberInColumns = calculateMaxPixelChangesNumberInColumns(wMinBound, wMaxBound, maxPointsNumberInColumns);

		size_t maxPointsNumberInRows;
		double maxPixelChangesNumberInRows = calculateMaxPixelChangesNumberInRows(hMinBound, hMaxBound, maxPointsNumberInRows);

		double meanInRows = calcMeanInRows(hMinBound, hMaxBound);
		double meanInColumns = calcMeanInColumns(wMinBound, wMaxBound);
		size_t meanChangesInRows = calcMeanChangesInRows();

		//std::cerr << "class: " << classId << ", changes: " << mineanChangesInRows << std::endl;
		// static size_t badCount = 0;
		// if ((maxPointsNumberInRows > imageH - 4) ^ (classId == 1))
		// {
		// 	++badCount;
		// 	std::cerr << maxPointsNumberInRows << std::endl;
		// 	std::cerr << badCount << ": " << classId << std::endl;
		// }

		// featureVector.push_back(meanInRows);
		// featureVector.push_back(meanInColumns);
		featureVector.push_back(meanChangesInRows);

		featureVector.push_back(maxPointsNumberInRows);
		featureVector.push_back(maxPointsNumberInColumns);

		int whiteComponentsNumber = findWhiteComponentsNumber();
		// std::cerr << "class: " << classId << ", components: " << whiteComponentsNumber << std::endl;

			// for (int i = 1; i < 5; ++i) {
		// 	featureVector.push_back(whiteComponentsNumber == i);
		// }
		featureVector.push_back(whiteComponentsNumber);

		// int blackComponentsNumber = findBlackComponentsNumber();
		// std::cerr << "class: " << classId << ", components: " << blackComponentsNumber << std::endl;

		// featureVector.push_back(blackComponentsNumber);

		/*
		partHNumber = 5;
		partWNumber = 5;
		partHeight = (imageH + partHNumber - 1) / partHNumber;
		partWidth = (imageW + partWNumber - 1) / partWNumber;

		for (int i = 0; i < partHNumber; ++i)
		{
			for (int j = 0; j < partWNumber; ++j)
			{
				int hMin = min(hMinBound + i * partHeight, H);
				int hMax = min(hMinBound + (i + 1) * partHeight, H);
				int wMin = min(wMinBound + j * partWidth, W);
				int wMax = min(wMinBound + (j + 1) * partWidth, W);

				featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
			}
		}
		*/

		partHNumber = 1;
		partWNumber = 3;
		partHeight = (imageH + partHNumber - 1) / partHNumber;
		partWidth = (imageW + partWNumber - 1) / partWNumber;

		for (int i = 0; i < partHNumber; ++i)
		{
			for (int j = 0; j < partWNumber; ++j)
			{
				int hMin = min(hMinBound + i * partHeight, H);
				int hMax = min(hMinBound + (i + 1) * partHeight, H);
				int wMin = min(wMinBound + j * partWidth, W);
				int wMax = min(wMinBound + (j + 1) * partWidth, W);

				featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
			}
		}

		partHNumber = 3;
		partWNumber = 1;
		partHeight = (imageH + partHNumber - 1) / partHNumber;
		partWidth = (imageW + partWNumber - 1) / partWNumber;

		for (int i = 0; i < partHNumber; ++i)
		{
			for (int j = 0; j < partWNumber; ++j)
			{
				int hMin = min(hMinBound + i * partHeight, H);
				int hMax = min(hMinBound + (i + 1) * partHeight, H);
				int wMin = min(wMinBound + j * partWidth, W);
				int wMax = min(wMinBound + (j + 1) * partWidth, W);

				featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
			}
		}

		partHNumber = 2;
		partWNumber = 2;
		partHeight = (imageH + partHNumber - 1) / partHNumber;
		partWidth = (imageW + partWNumber - 1) / partWNumber;

		// for (int i = 0; i < partHNumber; ++i)
		// {
		// 	for (int j = 0; j < partWNumber; ++j)
		// 	{
		// 		int hMin = min(hMinBound + i * partHeight, H);
		// 		int hMax = min(hMinBound + (i + 1) * partHeight, H);
		// 		int wMin = min(wMinBound + j * partWidth, W);
		// 		int wMax = min(wMinBound + (j + 1) * partWidth, W);

		// 		std::vector<int> HOG = calculateHOGInRectangle_4(hMin, wMin, hMax, hMax);
		// 		// std::cerr << "class: " << classId << std::endl;
		// 		for (int i = 0; i < HOG.size(); ++i) {
		// 			// std::cerr << HOG[i] << " ";
		// 			featureVector.push_back(HOG[i]);
		// 		}
		// 		std::cerr << std::endl;

		// 		featureVector.push_back(calculateMeanColorInSquare(hMin, wMin, hMax, wMax));
		// 	}
		// }

		std::vector<int> HOG = calculateHOGInRectangle_4(hMinBound, wMinBound, hMaxBound, wMaxBound);
		// std::cerr << "class: " << classId << std::endl;
		for (int i = 0; i < HOG.size(); ++i) {
			// std::cerr << HOG[i] << " ";
			featureVector.push_back(HOG[i]);
		}
		// std::cerr << std::endl;

		if (isTraining)
		{
			featureVector.push_back(classId);
		}
		return featureVector;
	}

	bool isOnImage(size_t h, size_t w) const
	{
		if (h < 0 || w < 0 || h >= H || w >= W)
		{
			return false;
		}
		return true;
	}

	struct DisjoinSet {
		DisjoinSet(int n) {
			componentsNumber = n;
			for (int i = 0; i < n; ++i) {
				parents.push_back(i);
			}
			sizes.resize(n, 1);
		}

		int findParent(int x) {
			if (parents[x] != x)
				parents[x] = findParent(parents[x]);
			return parents[x];
		}

		bool unite(int x, int y) {
			x = findParent(x);
			y = findParent(y);
			if (x == y)
				return false;
			if (rand() & 1)
				swap(x, y);
			parents[x] = y;
			sizes[y] += sizes[x];
			--componentsNumber;
			return true;
		}

		vector<int> parents;
		vector<int> sizes;
		int componentsNumber;
	};

	int findWhiteComponentsNumber() {
		DisjoinSet disjoinSet(H * W);

		int notPaintedCellsNumber = 0;

		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				int firstId = h * W + w;
				if (!isPainted(h, w)) 
				{
					++notPaintedCellsNumber;
					for (size_t k = 0; k < 4; ++k)
					{
						size_t nh = h + dh[k];
						size_t nw = w + dw[k];
						if (isOnImage(nh, nw) && !isPainted(nh, nw))
						{
							int secondId = nh * W + nw;
							disjoinSet.unite(firstId, secondId);
						}
					}
				}
			}
		}

		int paintedCellsNumber = H * W - notPaintedCellsNumber;
		std::unordered_set<int> processed;
		std::vector<int> sizes;
		for (int i = 0; i < H * W; ++i) {
			int x = disjoinSet.findParent(i);
			if (processed.find(x) == processed.end()) {
				processed.insert(x);
				if (disjoinSet.sizes[x] > 4) {
					sizes.push_back(disjoinSet.sizes[x]);
				}
			}
		}

		for (int i = 0; i < sizes.size(); ++i) {
			std::cerr << sizes[i] << " ";
		}
		std::cerr << endl;

		return sizes.size();
	}

	int findBlackComponentsNumber() {
		DisjoinSet disjoinSet(H * W);

		int paintedCellsNumber = 0;

		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				int firstId = h * W + w;
				if (isPainted(h, w)) 
				{
					++paintedCellsNumber;
					for (size_t k = 0; k < 4; ++k)
					{
						size_t nh = h + dh[k];
						size_t nw = w + dw[k];
						if (isOnImage(nh, nw) && isPainted(nh, nw))
						{
							int secondId = nh * W + nw;
							disjoinSet.unite(firstId, secondId);
						}
					}
				}
			}
		}
		int notPaintedCellsNumber = H * W - paintedCellsNumber;

		return disjoinSet.componentsNumber - notPaintedCellsNumber;
	}

	size_t calculatePointsNumberInCircle(double hCenter, double wCenter, double R)
	{
		size_t pointsNumber = 0;
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				if (isPainted(h, w)) 
				{
					if (distance(h, w, hCenter, wCenter) < R)
					{
						++pointsNumber;
					}
				}
			}
		}
		return pointsNumber;
	}

	void calculateMassCenter()
	{
		hMassCenter = 0;
		wMassCenter = 0;
		totalMass = 0;

		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				hMassCenter += h * image[h][w];
				wMassCenter += w * image[h][w];
				totalMass += image[h][w];
			}
		}
		hMassCenter /= totalMass;
		wMassCenter /= totalMass;
	}

	double calculateMeanDistanceToMassCenter()
	{
		double meanDistance = 0;
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				meanDistance += distance(h, w, hMassCenter, wMassCenter) * image[h][w];
			}
		}
		return meanDistance /= totalMass;
	}

	double calculateMeanDistanceToMassCenterPainted()
	{
		double meanDistance = 0;
		size_t paintedNumber = 0;
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				if (isPainted(h, w))
				{
					meanDistance += distance(h, w, hMassCenter, wMassCenter);
					++paintedNumber;
				}
			}
		}
		return meanDistance / paintedNumber;
	}

	size_t calcMeanChangesInRows()
	{
		double meanInRows = calcMeanInRows(hMinBound, hMaxBound);
		bool previousMeanWasBigger = false;
		size_t changesNumber = 0;

		for (int h = 0; h < H; ++h)
		{
			double mean = calcMeanInRow(h);
			if ((mean > meanInRows + 0.01) ^ (previousMeanWasBigger))
			{
				++changesNumber;
				previousMeanWasBigger ^= 1;
			}
		}

		return changesNumber;
	}

	double calculateMeanPixelChangesNumberInColumns(size_t firstColumn, size_t lastColumn)
	{
		if (firstColumn >= lastColumn)
		{
			return 0.0;
		}

		double meanPixelChangesNumber = 0.0;
		for (size_t w = firstColumn; w < lastColumn; ++w)
		{
			meanPixelChangesNumber += calculatePixelChangesNumberInColumn(w);
		}
		meanPixelChangesNumber /= lastColumn - firstColumn;
		return meanPixelChangesNumber;
	}

	size_t calculateMaxPixelChangesNumberInColumns(size_t firstColumn, size_t lastColumn, size_t& maxPointsNumber)
	{
		if (firstColumn >= lastColumn)
		{
			return 0.0;
		}

		size_t maxPixelChangesNumber = 0;
		maxPointsNumber = 0;
		for (size_t w = firstColumn; w < lastColumn; ++w)
		{
			size_t pixelChangesNumber = calculatePixelChangesNumberInColumn(w);
			if (maxPixelChangesNumber <= pixelChangesNumber) 
			{
				if (maxPixelChangesNumber == pixelChangesNumber)
				{
					++maxPointsNumber;
				}
				else
				{
					maxPixelChangesNumber = pixelChangesNumber;
					maxPointsNumber = 1;
				}
			}
		}
		return maxPixelChangesNumber;
	}

	double calculateMaxPixelChangesNumberInRows(size_t firstRow, size_t lastRow, size_t& maxPointsNumber)
	{
		if (firstRow >= lastRow)
		{
			return 0.0;
		}

		size_t maxPixelChangesNumber = 0;
		for (size_t h = firstRow; h < lastRow; ++h)
		{
			size_t pixelChangesNumber = calculatePixelChangesNumberInRow(h);
			if (maxPixelChangesNumber <= pixelChangesNumber) 
			{
				if (maxPixelChangesNumber == pixelChangesNumber)
				{
					++maxPointsNumber;
				}
				else
				{
					maxPixelChangesNumber = pixelChangesNumber;
					maxPointsNumber = 1;
				}
			}
		}
		return maxPixelChangesNumber;
	}

	size_t calculatePixelChangesNumberInColumn(size_t columnNumber)
	{
		bool previousPixelPainted = false;
		size_t pixelChangesNumber = 0;
		for (int h = 0; h < H; ++h)
		{
			if (isPainted(h, columnNumber) ^ previousPixelPainted)
			{
				previousPixelPainted ^= 1;
				++pixelChangesNumber;
			}
		}
		return pixelChangesNumber;
	}

	double calculateMeanPixelChangesNumberInRows(size_t firstRow, size_t lastRow)
	{
		if (firstRow >= lastRow)
		{
			return 0.0;
		}

		double meanPixelChangesNumber = 0.0;
		for (size_t h = firstRow; h < lastRow; ++h)
		{
			meanPixelChangesNumber += calculatePixelChangesNumberInRow(h);
		}
		meanPixelChangesNumber /= lastRow - firstRow;
		return meanPixelChangesNumber;
	}

	size_t calculatePixelChangesNumberInRow(size_t rowNumber)
	{
		bool previousPixelPainted = false;
		size_t pixelChangesNumber = 0;
		for (int w = 0; w < W; ++w)
		{
			if (isPainted(rowNumber, w) ^ previousPixelPainted)
			{
				previousPixelPainted ^= 1;
				++pixelChangesNumber;
			}
		}
		return pixelChangesNumber;
	}

	size_t calculatePerimeter() const
	{
		size_t perimeter = 0;
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				bool isOnBorder = false;
				if (isPainted(h, w)) 
				{
					for (size_t k = 0; k < 4; ++k)
					{
						size_t nh = h + dh[k];
						size_t nw = w + dw[k];
						if (isOnImage(nh, nw) && !isPainted(nh, nw))
						{
							isOnBorder = true;
						}
					}
				}
				perimeter += isOnBorder;
			}
		}
		return perimeter;
	}

	bool isPainted(size_t h, size_t w) const
	{
		return image[h][w] > 128;
	}

	void calculateBoundingBox()
	{
		hMinBound = wMinBound = H + W;
		hMaxBound = wMaxBound = 0;

		for (size_t h = 0; h < H; ++h)
		{
			for (size_t w = 0; w < W; ++w)
			{
				if (isPainted(h, w))
				{
					hMinBound = min(hMinBound, h);
					wMinBound = min(wMinBound, w);
					hMaxBound = max(hMaxBound, h);
					wMaxBound = max(wMaxBound, w);
				}
			}
		}
	}

	double calculateMeanColorInSquare(int hMin, int wMin, int hMax, int wMax)
	{
		if (hMin >= hMax || wMin >= wMax)
		{
			return 0.0;
		}

		double meanColor = 0;
		for (int h = hMin; h < hMax; ++h)
		{
			for (int w = wMin; w < wMax; ++w)
			{
				meanColor += image[h][w];
			}
		}
		meanColor /= (wMax - wMin) * (hMax - hMin);
		return meanColor;
	}

	std::vector<int> calculateHOGInRectangle_4(int hMin, int wMin, int hMax, int wMax) {
		std::vector<int> HOGDistribution(4);

		for (int h = hMin; h < hMax; ++h) {
			for (int w = wMin; w < wMax; ++w) {
				
				int maxGrad = -10000000;
				int maxK = 0;

				for (int k = 0; k < 4; ++k) {
					int nh = h + dh[k];
					int nw = w + dw[k];

					if (isOnImage(nh, nw)) {
						int diff = image[nh][nw] - image[h][w];
						if (diff > maxGrad) {
							maxGrad = diff;
							maxK = k;
						}
					}
				}

				++HOGDistribution[maxK];
			}
		}

		return HOGDistribution;
	}

	std::vector<int> calculateHOGInRectangle_9(int hMin, int wMin, int hMax, int wMax) {
		std::vector<int> HOGDistribution(9);

		for (int h = hMin; h < hMax; ++h) {
			for (int w = wMin; w < wMax; ++w) {
				
				int maxGrad = 0;
				int maxdH = 0;
				int maxdW = 0;

				for (int dH = -1; dH <= 1; ++dH) {
					for (int dW = -1; dW <= 1; ++dW) {
						int nh = h + dH;
						int nw = w + dW;

						if (isOnImage(nh, nw)) {
							int diff = image[nh][nw] - image[h][w];
							if (diff > maxGrad) {
								maxGrad = diff;
								maxdH = dH;
								maxdW = dW;
							}
						}
					}
				}

				int n = (maxdH + 1) * 3 + (maxdW + 1);
				++HOGDistribution[n];
			}
		}

		return HOGDistribution;
	}

	double calcMeanInRow(size_t rowNumber)
	{
		double mean = 0;
		for (int w = 0; w < W; ++w)
		{
			mean += image[rowNumber][w];
		}
		mean /= W;
		return mean;
	}

	double calcMeanInRows(size_t firstRow, size_t lastRow)
	{
		if (firstRow >= lastRow)
		{
			return 0.0;
		}

		double mean = 0;
		for (int h = firstRow; h < lastRow; ++h)
		{
			mean += calcMeanInRow(h);
		}
		mean /= lastRow - firstRow;
		return mean;
	}

	size_t calcMinInRow(size_t rowNumber)
	{
		for (int w = 0; w < W; ++w)
		{
			if (isPainted(rowNumber, w))
			{
				return w;
			}
		}
		return W;
	}

	size_t calcMaxInRow(size_t rowNumber)
	{
		for (int w = W - 1; w >= 0; --w)
		{
			if (isPainted(rowNumber, w))
			{
				return w;
			}
		}
		return -1;
	}

	size_t calcMinInColumn(size_t columnNumber)
	{
		for (int h = 0; h < H; ++h)
		{
			if (isPainted(h, columnNumber))
			{
				return h;
			}
		}
		return H;
	}

	size_t calcMaxInColumn(size_t columnNumber)
	{
		for (int h = H - 1; h >= 0; --h)
		{
			if (isPainted(h, columnNumber))
			{
				return h;
			}
		}
		return -1;
	}

	double calcMeanInColumn(size_t columnNumber)
	{
		double mean = 0;
		for (int h = 0; h < H; ++h)
		{
			mean += image[h][columnNumber];
		}
		mean /= H;
		return mean;
	}

	double calcMeanInColumns(size_t firstColumn, size_t lastColumn)
	{
		if (firstColumn >= lastColumn)
		{
			return 0.0;
		}

		double mean = 0;
		for (int w = firstColumn; w < lastColumn; ++w)
		{
			mean += calcMeanInColumn(w);
		}
		mean /= lastColumn - firstColumn;
		return mean;
	}

	size_t hMinBound, hMaxBound, wMinBound, wMaxBound;
	double hMassCenter, wMassCenter;
	double totalMass;

	Image image;
	size_t classId;
};

bool getImage(std::ifstream& in, Image& image, 
							size_t& classId)
{
	image = std::vector< std::vector<size_t> > (H, std::vector<size_t>(W, 0));

	if (isTraining)
	{ 
		in >> classId;
	}

	for (int h = 0; h < H; ++h)
	{
		for (int w = 0; w < W; ++w)
		{
			char c;
			if ((h != 0) || (w != 0) || isTraining) {
				in >> c;
			}
			in >> (image)[h][w];
		}
	}
}

std::vector<TrainObject> readTrainData(std::string fileName)
{
	std::ifstream in(fileName);

	std::string line;
	getline(in, line);

	std::vector<TrainObject> trainObjects;

	while (!in.eof())
	{
		TrainObject trainObject;
		getImage(in, trainObject.image, trainObject.classId);
		trainObjects.push_back(trainObject);
		getline(in, line);
	}
	trainObjects.resize(min(trainObjects.size(), samplesNumber));
	std::cerr << trainObjects.size() << " samples" << std::endl;
	return trainObjects;
}

void buildFeatureVectors(std::vector<TrainObject>& trainObjects, 
													std::string fileName)
{
	std::ofstream out(fileName);
	size_t featureVectorSize = trainObjects[0].buildFeatureVector().size();

	for (size_t feature = 1; feature < featureVectorSize; ++feature)
	{
		out << "F" << feature << ",";
	}
	if (isTraining)
	{
		out << "label" << std::endl;
	}
	else
	{
		out << "F" << featureVectorSize << std::endl;
	}

	for (auto& trainObject : trainObjects)
	{
		std::vector<double> featureVector = trainObject.buildFeatureVector();
		for (size_t feature = 0; feature < featureVectorSize - 1; ++feature)
		{
			out << featureVector[feature] << ",";
		}
		out << featureVector[featureVectorSize - 1] << std::endl;
	}
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		printf("Usage: %s filename isTraining [samplesNumber]\n", argv[0]);
		return 0;
	}
	std::string filename = argv[1];
	isTraining = atoi(argv[2]);

	if (argc > 3)
	{
		samplesNumber = atoi(argv[3]);
	}
	std::vector<TrainObject> trainObjects = readTrainData(filename);
	buildFeatureVectors(trainObjects, filename + ".processed");
	return 0;
}
