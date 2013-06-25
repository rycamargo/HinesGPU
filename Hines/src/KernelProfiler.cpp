/*
 * KernelProfiler.cpp
 *
 *  Created on: 17/07/2012
 *      Author: rcamargo
 */

#include "KernelProfiler.hpp"
#include "PlatformFunctions.hpp"

KernelProfiler::KernelProfiler(int nKernelTypes, int nThreads, int nInputIndexes, int currProcess) {

    char buf[20];
    sprintf(buf, "%s%d%s", "profiler", currProcess, ".dat");
    this->profileFile = fopen(buf, "a");

	this->nKernelTypes  = nKernelTypes;
	this->nThreads      = nThreads;
	this->nInputIndexes = nInputIndexes;

	profilerStartTmp  = new uint64 **[nKernelTypes];
	meanProfilerTimes = new uint64 **[nKernelTypes];
	profileInfo       = new int    **[nKernelTypes];
	for (int i=0; i<nKernelTypes; i++) {
		profilerStartTmp[i]  = new uint64 *[nThreads];
		meanProfilerTimes[i] = new uint64 *[nThreads];
		profileInfo[i]       = new int    *[nThreads];
		for (int j=0; j<nThreads; j++) {
			profilerStartTmp[i][j]  = new uint64[nInputIndexes];
			meanProfilerTimes[i][j] = new uint64[nInputIndexes];
			profileInfo[i][j]       = new int   [nInputIndexes];
			for (int k=0; k<nInputIndexes; k++) {
				profilerStartTmp[i][j][k]  = 0;
				meanProfilerTimes[i][j][k] = 0;
				profileInfo[i][j][k]       = 0;
			}
		}
	}

}

KernelProfiler::~KernelProfiler() {

	for (int i=0; i<nKernelTypes; i++) {
		for (int j=0; j<nThreads; j++) {
			delete[] profilerStartTmp[i][j];
			delete[] meanProfilerTimes[i][j];
			delete[] profileInfo[i][j];
		}
		delete[] profilerStartTmp[i];
		delete[] meanProfilerTimes[i];
		delete[] profileInfo[i];
	}
	delete[] profilerStartTmp;
	delete[] meanProfilerTimes;
	delete[] profileInfo;
}

void KernelProfiler::setProfileInfo (int thread, int inputIndex, int kernelType, int info){
	profileInfo[kernelType][thread][inputIndex] = info;
}

void KernelProfiler::setKernelStart(int thread, int inputIndex, int kernelType) {
	profilerStartTmp[kernelType][thread][inputIndex] = gettimeInMilli();
}

void KernelProfiler::setKernelFinish(int thread, int inputIndex, int kernelType) {

	meanProfilerTimes[kernelType][thread][inputIndex] +=
			gettimeInMilli() - profilerStartTmp[kernelType][thread][inputIndex];

}

void KernelProfiler::printProfile(int kernelType) {

	if (kernelType >= 0) {
		for(int t=0; t<nThreads; t++)
			for(int i=0; i<nInputIndexes; i++)
				if (meanProfilerTimes[kernelType][t][i] != 0)
					printf("thread=%1d,type=%1d -> %-10.5f\n", t, i, meanProfilerTimes[kernelType][t][i] / 1000.);
	}
	else {
		for(int t=0; t<nThreads; t++) {
			ftype sum = 0;
			for(int i=0; i<nInputIndexes; i++)
				for(int k=0; k<nKernelTypes; k++)
					sum += meanProfilerTimes[k][t][i];
			printf("thread=%1d -> %-10.5f\n", t, sum / 1000.);
		}
	}
}

/**
 * nTypesTotal=3 nConnPerNeuron=100 rateLevel=l
 * type=0 kernel=0 count=7000 time=54.5
 * type=0 kernel=0 count=3000 time=28.5
 * type=1 kernel=0 count=1000 time=13.1
 * type=0 kernel=1 count=7000 time=2.1
 * type=0 kernel=1 count=3000 time=1.2
 * type=1 kernel=1 count=1000 time=0.6
 * #----------------------------------------------------------
 */

void KernelProfiler::printProfileToFile(int nNeurons, int nProcesses, int nTypesTotal, int nConnPerNeuron, int rateLevel) {

	fprintf(profileFile, "nNeurons=%d nProcesses=%d nTypesTotal=%1d nConnPerNeuron=%-5d rateLevel=%1d\n",
			nNeurons, nProcesses, nTypesTotal, nConnPerNeuron, rateLevel);

	for (int k=0; k<nKernelTypes; k++) {
		for(int t=0; t<nThreads; t++)
			for(int i=0; i<nInputIndexes; i++)
				if (meanProfilerTimes[k][t][i] != 0)
					fprintf(profileFile, "type=%1d kernel=%d count=%-6d time=%-10.5f\n",
							i, k, profileInfo[k][t][i], meanProfilerTimes[k][t][i] / 1000.);
	}
	fprintf (profileFile, "#------------------------------------------------------------------------------\n");
}

