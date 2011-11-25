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

	int *groupList;
	int *neuronList;

public:
	NeuronInfoWriter(ThreadInfo *tInfo);
	~NeuronInfoWriter();

	//void setMonitoredList( int nMonitored, int *groupList, int *neuronList );

    void writeVmToFile(int kStep);

    void updateSampleVm(int kStep);
    void writeSampleVm(int kStep);

    void writeResultsToFile(char mode, int nNeuronsTotal, int nComp, BenchTimes & bench);
};

#endif /* NEURONINFOWRITER_H_ */
