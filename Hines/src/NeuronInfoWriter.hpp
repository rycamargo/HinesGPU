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
	FILE *outFile;
	FILE *vmKernelFile;
	int nVmTimeSeries;

public:
	NeuronInfoWriter(ThreadInfo *tInfo);
	~NeuronInfoWriter();

    void writeVmToFile(int kStep);
    void writeSampleVm(int kStep);
};

#endif /* NEURONINFOWRITER_H_ */
