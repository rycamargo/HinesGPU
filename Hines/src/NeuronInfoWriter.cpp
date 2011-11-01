#include <cstdio>
#include "NeuronInfoWriter.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

NeuronInfoWriter::NeuronInfoWriter(ThreadInfo *tInfo) {

	this->tInfo = tInfo;
	this->sharedData = tInfo->sharedData;
	this->kernelInfo = tInfo->sharedData->kernelInfo;

    char buf[20];

    this->outFile;
    sprintf(buf, "%s%d%s", "sampleVm", tInfo->currProcess, ".dat");
    this->outFile = fopen(buf, "w");

    this->vmKernelFile;
    sprintf(buf, "%s%d%s", "vmKernel", tInfo->currProcess, ".dat");
    this->vmKernelFile = fopen(buf, "w");

    this->nVmTimeSeries = 4;
    int nCompVmTimeSerie = sharedData->matrixList[tInfo->startTypeThread][0].nComp;

    this->vmTimeSerie = (ftype**)(((malloc(sizeof (ftype*) * this->nVmTimeSeries))));

    this->vmTimeSerieMemSize = sizeof (ftype) * (nCompVmTimeSerie * kernelInfo->nKernelSteps);
    for(int k = 0;k < nVmTimeSeries;k++)
    	this->vmTimeSerie[k] = (ftype*)(((malloc(vmTimeSerieMemSize))));

}

NeuronInfoWriter::~NeuronInfoWriter () {

	fclose(outFile);
	fclose(vmKernelFile);

	for(int k = 0;k < nVmTimeSeries; k++)
		free( vmTimeSerie[k] );

	free (vmTimeSerie);

}

void NeuronInfoWriter::writeVmToFile(int kStep) {

    for(int type = tInfo->startTypeProcess; type < tInfo->endTypeProcess;type++){
        fprintf(vmKernelFile, "dt=%-10.2f\ttype=%d\t", sharedData->dt * (kStep + kernelInfo->nKernelSteps), type);
        for(int n = 0;n < tInfo->nNeurons[type];n++)
            fprintf(vmKernelFile, "%10.2f\t", sharedData->synData->vmListHost[type][n]);

        fprintf(vmKernelFile, "\n");
    }
}

void NeuronInfoWriter::writeSampleVm(int kStep)
{
    if(benchConf.verbose == 1)
        printf("Writing Sample Vms thread=%d\n", tInfo->threadNumber);

    int t1 = tInfo->startTypeThread, n1 = 0;
    if(tInfo->startTypeThread <= t1 && t1 < tInfo->endTypeThread)
        cudaMemcpy(vmTimeSerie[0], sharedData->hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

    t1 = tInfo->startTypeThread;
    n1 = 1; //2291;
    if(tInfo->startTypeThread <= t1 && t1 < tInfo->endTypeThread)
        cudaMemcpy(vmTimeSerie[1], sharedData->hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

    t1 = tInfo->endTypeThread - 1;
    n1 = 2; //135;
    if(tInfo->startTypeThread <= t1 && t1 < tInfo->endTypeThread)
        cudaMemcpy(vmTimeSerie[2], sharedData->hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

    t1 = tInfo->endTypeThread - 1;
    n1 = 3; //1203;
    if(tInfo->startTypeThread <= t1 && t1 < tInfo->endTypeThread)
        cudaMemcpy(vmTimeSerie[3], sharedData->hList[t1][n1].vmTimeSerie, vmTimeSerieMemSize, cudaMemcpyDeviceToHost);

    for(int i = kStep;i < kStep + kernelInfo->nKernelSteps; i++){
        fprintf(outFile, "%10.2f\t%10.2f\t%10.2f\t%10.2f\t%10.2f\n", sharedData->dt * (i + 1), vmTimeSerie[0][(i - kStep)], vmTimeSerie[1][(i - kStep)], vmTimeSerie[2][(i - kStep)], vmTimeSerie[3][(i - kStep)]);
    }
}
