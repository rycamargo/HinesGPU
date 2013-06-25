/*
 * KernelProfiler.hpp
 *
 *  Created on: 17/07/2012
 *      Author: rcamargo
 */

#ifndef KERNELPROFILER_HPP_
#define KERNELPROFILER_HPP_

#define PROFILER_HINES 0
#define PROFILER_COMM 1

#include "Definitions.hpp"
#include <cstdio>

class KernelProfiler {

	int nKernelTypes;
	int nThreads;
	int nInputIndexes;

	uint64*** meanProfilerTimes;

	uint64*** profilerStartTmp;

	int*** profileInfo;

	FILE *profileFile;

public:
	KernelProfiler(int nKernelTypes, int nThreads, int nInputIndexes, int currProcess);
	virtual ~KernelProfiler();

	void setProfileInfo (int thread, int inputIndex, int kernelType, int info);

	void setKernelStart (int thread, int inputIndex, int kernelType);
	void setKernelFinish(int thread, int inputIndex, int kernelType);

	void printProfile(int kernelType);
	void printProfileToFile(int nNeurons, int nProcesses, int nTypesTotal, int nConnPerNeuron, int rateLevel);

};

#endif /* KERNELPROFILER_HPP_ */
