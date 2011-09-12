/*
 * SpikeStatistics.hpp
 *
 *  Created on: 10/09/2009
 *      Author: rcamargo
 */

#ifndef SPIKESTATISTICS_HPP_
#define SPIKESTATISTICS_HPP_

#include "PlatformFunctions.hpp"
#include <cstdio>

class SpikeStatistics {

	int *nNeurons;
	int *typeList;
	int nTypes;

	int totalNeurons;
	int pyrNeurons;
	int inhNeurons;

	double **totalGeneratedSpikes;
	double **totalReceivedSpikes;

	ftype **lastGeneratedSpikeTimes;

public:
	SpikeStatistics(int *nNeurons, int nTypes, int *typeList);
	virtual ~SpikeStatistics();

	void addGeneratedSpikes(int type, int neuron, ftype *spikeTimes, int nSpikes);

	void addReceivedSpikes(int type, int neuron, int nReceivedSpikes);

	void printSpikeStatistics(char *filename, ftype currentTime, BenchTimes & bench);

	void printKernelSpikeStatistics(FILE *nSpkfile, FILE *lastSpkfile, ftype currentTime);
};

#endif /* SPIKESTATISTICS_HPP_ */
