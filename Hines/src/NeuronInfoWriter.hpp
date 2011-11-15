#include "HinesStruct.hpp"

#ifndef NEURONINFOWRITER_H_
#define NEURONINFOWRITER_H_

class NeuronInfoWriter {

private:
	ThreadInfo *tInfo;
	SharedNeuronGpuData *sharedData;
	KernelInfo *kernelInfo;

	ftype **vmTimeSerie;
	int vmTimeSerieMemSize;
	int nVmTimeSeries;

	FILE *outFile;
	FILE *vmKernelFile;
	FILE *resultFile;

public:
	NeuronInfoWriter(ThreadInfo *tInfo);
	~NeuronInfoWriter();

    void writeVmToFile(int kStep);
    void writeSampleVm(int kStep);
    void writeResultsToFile(char mode, int nNeuronsTotal, int nComp, BenchTimes & bench);
};

#endif /* NEURONINFOWRITER_H_ */
