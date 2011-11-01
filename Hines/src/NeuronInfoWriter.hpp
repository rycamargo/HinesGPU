#include "HinesStruct.hpp"

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
