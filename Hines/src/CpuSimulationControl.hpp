#ifndef CPUSIMULATIONCONTROL_H_
#define CPUSIMULATIONCONTROL_H_

class CpuSimulationControl {

private:
	struct ThreadInfo * tInfo;
	struct SharedNeuronGpuData *sharedData;
	struct KernelInfo *kernelInfo;

public:
	CpuSimulationControl(struct ThreadInfo *tInfo);
	//int launchGpuExecution();

//private:

    void performCpuNeuronalProcessing();
    void addReceivedSpikesToTargetChannelCPU();
};

#endif /* CPUSIMULATIONCONTROL_H_ */
