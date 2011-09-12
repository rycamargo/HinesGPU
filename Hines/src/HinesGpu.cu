/*
 * HinesGpu.cu
 *
 *  Created on: 06/06/2009
 *      Author: rcamargo
 */

/**
 * TODO: otimizar
 * - acessar sTriangList com POS
 * - Ver acessos à memória global
 */

/**
 * TODO:
 * - Implementar neurônios de minha simulação
 * - Implementar rede com múltiplos neurônios
 *
 * - Testar desempenho e comparar com GENESIS
 *
 * - Otimizar (usar memória compartilhada)
 * - Otimizar (reduzir número de alocações de memória para copiar lista de spikes)
 */

//extern "C" {
#include "HinesMatrix.hpp"
#include "PlatformFunctions.hpp"
#include "HinesStruct.hpp"
#include <cassert>

#include <cuda.h> // Necessary to allow better eclipse integration
#include <cuda_runtime_api.h> // Necessary to allow better eclipse integration
#include <device_launch_parameters.h> // Necessary to allow better eclipse integration
#include <device_functions.h> // Necessary to allow better eclipse integration

//#define POS(i) (i) + nComp*threadIdx.x
#define POS(i) (i)*blockDim.x+threadIdx.x
//#define POS1(i) (i) + leftListSize*threadIdx.x
#define POS1(i) (i)*blockDim.x + threadIdx.x

#define NLOCALSPIKES 16 // 16

__device__ void printMatrix(HinesStruct *hList, ftype *list, ftype *rhsLocal, ftype *vmListLocal) {


	/*
	printf ("----------------------------------------------------------\n");
	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];
	int nComp = h.nComp;

	int pos = 0;
	ftype zero = 0;
	for (int i=0; i<nComp; i++) {
		for (int j=0; j<nComp; j++) {

			if (i == h.leftListLine[pos] && j == h.leftListColumn[pos])
				printf( "%10.4e\t", list[pos++]);
			else
				printf( "%10.4e\t", zero);
		}
		printf( "%10.2f\t", vmListLocal[POS(i)]); // mV
		printf( "%10.2e\t\n", rhsLocal[POS(i)]);
	}
	*/
}

/***************************************************************************
 * This part is executed in every integration step
 ***************************************************************************/

__device__ ftype getCurrent(ftype vm, ucomp synType, ftype spikeTime, ftype weight,
		ftype currTime, ftype *sTau, ftype *sGmax, ftype *sEsyn) {

	ftype gsyn = 0;
	ftype current = 0;
	if (synType == SYNAPSE_AMPA) {
		ftype r = (currTime - spikeTime) / sTau[2*SYNAPSE_AMPA];
		gsyn = sGmax[SYNAPSE_AMPA] * r * expf(1 - r) * weight;
		//current = (vmListLocal[POS(comp)] - sEsyn[SYNAPSE_AMPA]) * gsyn;
		current = (vm - sEsyn[SYNAPSE_AMPA]) * gsyn;
	}

	else if (synType == SYNAPSE_GABA) {
		ftype r = (currTime - spikeTime) / sTau[2*SYNAPSE_GABA];
		gsyn = sGmax[SYNAPSE_GABA] * r * expf(1 - r) * weight;
		//current = (vmListLocal[POS(comp)] - sEsyn[SYNAPSE_GABA]) * gsyn;
		current = (vm - sEsyn[SYNAPSE_GABA]) * gsyn;
	}

	return current;
}

__device__ void findSynapticCurrentsNew(HinesStruct *hList, ftype *active, ftype *vmListLocal,
		ftype currTime, ftype *sTau, ftype *sGmax, ftype *sEsyn,ftype* freemem) {

	ftype *spkTimes, *spkWeigths;
	ftype *currentList = freemem;

	for (int neuron = 0 ; neuron < blockDim.x; neuron++) {

		HinesStruct & h = hList[neuron + blockIdx.x * blockDim.x];

		int synapseListSize = h.synapseListSize;
		int spikeListSize   = h.spikeListSize;
		if (spikeListSize == 0) continue; // skip neurons with no spikes

		ftype *spikeList = h.spikeList;
		ftype *synapseWeightList = h.synapseWeightList;

		for (int syn=0; syn < synapseListSize; syn++) {

			int synComp = h.synapseCompList[syn];
			int lastSpikePos = spikeListSize;
			if ( syn < (synapseListSize-1) )
				lastSpikePos = h.synSpikeListPos[syn+1];
			ucomp synapseType = h.synapseTypeList[syn];

			int spike = h.synSpikeListPos[syn];

			currentList[threadIdx.x] = 0;
			ftype vm = vmListLocal[synComp * blockDim.x + neuron];
			while (spike < lastSpikePos) {

				// Reads data in coalesced mode
				if (spike + threadIdx.x < lastSpikePos) {
					spkTimes[threadIdx.x]   = spikeList[ spike + threadIdx.x ];
					spkWeigths[threadIdx.x] = synapseWeightList[ spike + threadIdx.x];

					// Evaluates the current for the selected synapse

					if (currTime >= spkTimes[threadIdx.x])
						currentList[threadIdx.x] += getCurrent(vm, synapseType,
								spkTimes[threadIdx.x], spkWeigths[threadIdx.x],
								currTime, sTau, sGmax, sEsyn);

				}
				spike += blockDim.x;
			}


			__syncthreads();

			if (threadIdx.x % 2 == 0)
				currentList[threadIdx.x + 1] += currentList[threadIdx.x]; // 32

			if (threadIdx.x % 4 == 0)
				currentList[threadIdx.x + 3] += currentList[threadIdx.x + 1]; // 16

			if (threadIdx.x % 8 == 0)
				currentList[threadIdx.x + 7] += currentList[threadIdx.x + 3]; // 8

			if (threadIdx.x % 16 == 0)
				currentList[threadIdx.x + 15] += currentList[threadIdx.x + 7]; // 4

			if (threadIdx.x % 32 == 0)
				currentList[threadIdx.x + 31] += currentList[threadIdx.x + 15]; // 2

			if (threadIdx.x == 0)
				active [synComp * blockDim.x + neuron] += (currentList[63] + currentList[31]);

			//if (threadIdx.x % 128 == 0)
			//	currentList[threadIdx.x + 127] += currentList[threadIdx.x + 63]; // 2

			//active [synComp * blockDim.x + neuron] += currentList[63];

			//__syncthreads();
		}
	}
}

__device__ void findSynapticCurrentsTest(HinesStruct *hList, ftype *active, ftype *vmListLocal,
		ftype currTime, ftype *sTau, ftype *sGmax, ftype *sEsyn, ftype* freemem) {

	//ftype *freeMem;

	int coalescingSize = 4;

	HinesStruct & h = hList[blockIdx.x * blockDim.x + threadIdx.x];

	int *spikeListSize   = (int *)freemem;
	spikeListSize[threadIdx.x]   = h.spikeListSize;

	int *lastSpikePos	 = spikeListSize + blockDim.x;
	int *spike	 		 = lastSpikePos + blockDim.x;

	ftype **spikeList = (ftype **)(spike + blockDim.x);
	spikeList[threadIdx.x] = h.spikeList;

	ftype **synapseWeightList = spikeList + blockDim.x;
	synapseWeightList[threadIdx.x] = h.synapseWeightList;

	int **synSpikeListPos = (int **)(synapseWeightList + blockDim.x);
	synSpikeListPos[threadIdx.x] = h.synSpikeListPos;

	ftype *spk = (ftype *)(synSpikeListPos + blockDim.x);
	ftype *w = (ftype *)(spk + coalescingSize * blockDim.x);

	int synapseListSize = h.synapseListSize;

	if (spikeListSize[threadIdx.x] > 0) {

		spike[threadIdx.x] = 0;
		for (int syn=0; syn < synapseListSize; syn++) {
			ucomp synComp = h.synapseCompList[syn];
			lastSpikePos[threadIdx.x] = spikeListSize[threadIdx.x];
			if ( syn < (synapseListSize-1) )
				lastSpikePos[threadIdx.x] = synSpikeListPos[threadIdx.x][syn+1];
			ucomp synapseType = h.synapseTypeList[syn];

			for (; spike[threadIdx.x] < lastSpikePos[threadIdx.x]; spike[threadIdx.x] += coalescingSize ) {

//				for (int i=0; i<coalescingSize; i++) {
//
//					int pos = coalescingSize*threadIdx.x + i;
//					if (spike[threadIdx.x] + i < lastSpikePos[threadIdx.x]) {
//						spk[ pos ] = spikeList[threadIdx.x][spike[threadIdx.x] + i];
//						w[ pos ]   = synapseWeightList[threadIdx.x][spike[threadIdx.x] + i];
//					}
//
//				}

				for (int i=0; i<coalescingSize; i++) {

					int neuronPos = (threadIdx.x - threadIdx.x % coalescingSize) + i;
					int spkPos = neuronPos * coalescingSize + threadIdx.x % coalescingSize;

					if (spike[neuronPos] + threadIdx.x % coalescingSize < lastSpikePos[neuronPos]) {
						spk[ spkPos ] = spikeList[neuronPos][spike[neuronPos] + threadIdx.x % coalescingSize];
						w[ spkPos ]   = synapseWeightList[neuronPos][spike[neuronPos] + threadIdx.x % coalescingSize];
					}
				}

				ftype current = 0;
				for (int i=0; i<coalescingSize; i++) {

					int pos = coalescingSize*threadIdx.x + i;
					if (currTime >= spk[pos])
						current += getCurrent(vmListLocal[synComp * blockDim.x + threadIdx.x],
							synapseType, spk[pos], w[pos], currTime, sTau, sGmax, sEsyn);
				}

				active [POS(synComp)] += current;
			}

			spike[threadIdx.x] = lastSpikePos[threadIdx.x];
		}
	}

}

__device__ void findSynapticCurrents(HinesStruct *hList, ftype *active, ftype *vmListLocal,
		ftype currTime, ftype *sTau, ftype *sGmax, ftype *sEsyn, ftype* freemem) {

	//ftype *freeMem;

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];
	int nNeurons = h.nNeurons;

	int synapseListSize = h.synapseListSize;
	int spikeListSize   = h.spikeListSize;

	ftype *spikeList = h.spikeList;
	ftype *synapseWeightList = h.synapseWeightList;

	if (spikeListSize > 0) {

		int spike = h.synSpikeListPos[0];
		for (int syn=0; syn < synapseListSize; syn++) {
			int synComp = h.synapseCompList[syn];
			int lastSpikePos = ( syn < (synapseListSize-1) ) ?
					h.synSpikeListPos[syn+1] : h.synSpikeListPos[0] + spikeListSize;
			ucomp synapseType = h.synapseTypeList[syn];

			for (; spike < lastSpikePos; spike += 2 ) {

				ftype spk1 = spikeList[neuron + spike * nNeurons];
				ftype spk2 = (spike+1 < lastSpikePos) ? spikeList[neuron + (spike+1) * nNeurons] : 1e100000;
				ftype w1   = synapseWeightList[neuron + spike * nNeurons];
				ftype w2   = (spike+1 < lastSpikePos) ? synapseWeightList[neuron + (spike+1) * nNeurons] : 0;

				ftype current = 0;
				if (currTime >= spk1)
					current += getCurrent(vmListLocal[synComp * blockDim.x + threadIdx.x],
							synapseType, spk1, w1, currTime, sTau, sGmax, sEsyn);
				if (currTime >= spk2)
					current += getCurrent(vmListLocal[synComp * blockDim.x + threadIdx.x],
							synapseType, spk2, w2, currTime, sTau, sGmax, sEsyn);

				active [POS(synComp)] += current;
			}


			spike = lastSpikePos;
		}
	}

}

/**
 * Find the gate openings in the next time step
 * m(t + dt) = a + b m(t - dt)
 */
__device__ void evaluateGatesG( HinesStruct *hList, ftype *vmListLocal,
		ftype *nGate, ftype *hGate, ftype *mGate ) {

	//int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[blockIdx.x * blockDim.x + threadIdx.x];

	ftype alpha, beta, a, b;
	ftype V;
	ftype dtRec = 1/h.dt;

	for (int i=0; i<h.compListSize; i++) {
		V = vmListLocal[ POS(h.compList[i]) ];

		// gate m
		alpha = (V != 25.0) ? (0.1 * (25 - V)) / ( expf( 0.1 * (25-V) ) - 1 ) : 1; // ____expff para double
		beta  = 4 * expf( -V/18 );
		a = alpha / (dtRec + (alpha + beta)/2);
		b = (dtRec - (alpha + beta)/2) / (dtRec + (alpha + beta)/2);
		mGate[POS(i)] = a + b * mGate[POS(i)];

		// gate h
		alpha =  0.07 * expf ( -V/20 );
		beta  = 1 / ( expf( (30-V)/10 ) + 1 );
		a = alpha / (dtRec + (alpha + beta)/2);
		b = (dtRec - (alpha + beta)/2) / (dtRec + (alpha + beta)/2);
		hGate[POS(i)] = a + b * hGate[POS(i)];

	 	// gate n
		alpha = (V != 10.0) ? (0.01 * (10 - V)) / ( expf( 0.1 * (10-V) ) - 1 ) : 0.1;
		beta  = 0.125 * expf ( -V/80 );
		a = alpha / (dtRec + (alpha + beta)/2);
		b = (dtRec - (alpha + beta)/2) / (dtRec + (alpha + beta)/2);
		nGate[POS(i)] = a + b * nGate[POS(i)];

	}
}

__device__ void findActiveCurrentsG(HinesStruct *hList, ftype *activeList, ftype *vmListLocal,
		ftype *nGate, ftype *hGate, ftype *mGate, int nComp) {

	evaluateGatesG(hList, vmListLocal, nGate, hGate, mGate);

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	/**
	 * Update the channel conductances
	 */
	ftype Ek  = h.EK;
	ftype ENa = h.ENa;
	for (int i=0; i<h.compListSize; i++) {
		ftype gNaChannel = h.gNaBar[i] * mGate[POS(i)] * mGate[POS(i)] * mGate[POS(i)] * hGate[POS(i)];
		ftype gKChannel  =  h.gKBar[i] * nGate[POS(i)] * nGate[POS(i)] * nGate[POS(i)] * nGate[POS(i)];

		int comp = h.compList[i];
		ftype active = - gNaChannel * ENa - gKChannel  * Ek  ;
		active -=  ( 1 / h.Rm[comp] ) * ( h.ELeak );

		activeList[ POS(comp) ] += active;
		h.gNaChannel[i] = gNaChannel;
		h.gKChannel[i] = gKChannel;
	}

}

__device__ void upperTriangularizeAll(HinesStruct *hList, ftype *sTriangList,
				ftype *sLeftList, ucomp *sLeftListLine, ucomp *sLeftListColumn,
				ucomp *sLeftStartPos, ftype *rhsLocal, ftype *vmListLocal,
				ftype *nGate, ftype *hGate, ftype *mGate,
				ftype *sTau, ftype *sGmax, ftype *sEsyn, ftype *freeMem) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	ftype *active = freeMem;
	ftype *Cm = h.Cm;
	ftype *curr = h.curr;

	int nComp = h.nComp;
	int leftListSize = h.leftListSize;

	for (int i=0; i<nComp; i++) {
		active[ POS(i)] = 0;
	}

	//__syncthreads();

	findActiveCurrentsG(hList, active, vmListLocal, nGate, hGate, mGate, nComp);

	findSynapticCurrents(hList, active, vmListLocal, h.currStep * h.dt, sTau, sGmax, sEsyn,
			active + blockDim.x * nComp );

//	//int neuron = blockIdx.x * blockDim.x + threadIdx.x;
//	if (h.type == 0 && neuron == 1) {
//		for (int i=0; i<nComp; i++)
//			printf("active=%f\n", active[POS(i)]);
//		printf("\n");
//	}


	ftype dtRec = 1/h.dt;
	//rhsLocal[POS(0)] = (-2) * vmListLocal[POS(0)] * Cm[0] * dtRec - curr[0] + active[POS(0)];
	for (int i=0; i<nComp; i++)
		rhsLocal[POS(i)] = (-2) * vmListLocal[POS(i)] * Cm[i] * dtRec - curr[i] + active[POS(i)];

	// ***
	// 1000ms 960 16 1 -> 0.125ms
	for (int k = 0; k < leftListSize; k++)
		sTriangList[k] = sLeftList[k];

	for (int i = 0; i < h.compListSize; i++) {

		int comp = h.compList[i];
		int pos = sLeftStartPos[ comp ];

		for (; sLeftListColumn[pos] < comp && pos < leftListSize ; pos++);

		sTriangList[pos] -= (h.gNaChannel[i] + h.gKChannel[i]);
	}


	// 1000ms 960 16 1 -> 0.640ms
	for (int k = 0; k < leftListSize; k++) {

		int c = sLeftListColumn[k];
		int l = sLeftListLine[k];

		if( c < l ) {

			int pos = sLeftStartPos[c];
			for (; c == sLeftListLine[pos]; pos++)
				if (sLeftListColumn[pos] == c)
					break;

			ftype mul = -sTriangList[k] / sTriangList[pos];

			pos = sLeftStartPos[c];
			int tempK = sLeftStartPos[l];

			for (; c == sLeftListLine[pos] && pos < leftListSize; pos++) {
				for (; sLeftListColumn[tempK] < sLeftListColumn[pos] && tempK < leftListSize ; tempK++);

				sTriangList[tempK] += sTriangList[pos] * mul;
			}
			rhsLocal[POS(l)] += rhsLocal[POS(c)] * mul;
		}
	}


}



__device__ void updateRhsG(HinesStruct *hList, 
						   ftype *sMulList, ucomp *sMulListComp, ucomp *sMulListDest,
						   ftype *rhsLocal, ftype *vmListLocal,
						   ftype *nGate, ftype *hGate, ftype *mGate,
						   ftype *sTau, ftype *sGmax, ftype *sEsyn,
						   ftype *freeMem) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	ftype *active = freeMem;
	ftype *Cm = h.Cm;
	ftype *curr = h.curr;

	int nComp = h.nComp;

	for (int i=0; i<nComp; i++) {
		active[ POS(i)] = 0;
	}

	//__syncthreads();

	findActiveCurrentsG(hList, active, vmListLocal, nGate, hGate, mGate, nComp);
	findSynapticCurrents(hList, active, vmListLocal, h.currStep * h.dt, sTau, sGmax, sEsyn,
			active + blockDim.x * nComp);

	ftype dtRec = 1/h.dt;
	for (int i=0; i<nComp; i++)
		rhsLocal[POS(i)] = (-2) * vmListLocal[POS(i)] * Cm[i] * dtRec - curr[i] + active[POS(i)];

	int mulListSize = h.mulListSize;
	for (int mulListPos = 0; mulListPos < mulListSize; mulListPos++) {
		int dest = sMulListDest[mulListPos];
		int pos  = sMulListComp[mulListPos];
		rhsLocal[POS(dest)] += rhsLocal[POS(pos)] * sMulList[mulListPos];
	}

}

__device__ void backSubstituteG(HinesStruct *hList, 
								ftype *sTriangList, ucomp *sLeftListLine, ucomp *sLeftListColumn, 
								ftype *rhsLocal, ftype *vmListLocal, ftype* freeMem) {

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	HinesStruct & h = hList[neuron];

	int nComp = h.nComp;
	//int index1;
	int leftListSize = h.leftListSize;

	ftype *vmTmpLocal = freeMem;

	//index1 = nComp-1;

	if (h.triangAll == 0 && h.compListSize > 0) // has active channels only in soma
		vmTmpLocal[POS(nComp-1)] = rhsLocal[POS(nComp-1)] / ( sTriangList[(leftListSize-1)] - h.gNaChannel[0] - h.gKChannel[0] );
	else
		vmTmpLocal[POS(nComp-1)] = rhsLocal[POS(nComp-1)] / sTriangList[(leftListSize-1)];


	ftype tmp = 0;
	for (int leftListPos = leftListSize-2; leftListPos >=0 ; leftListPos--) {
		int line   = sLeftListLine[(leftListPos)];
		int column = sLeftListColumn[(leftListPos)];
		if (line == column) {
			vmTmpLocal[POS(line)] = (rhsLocal[POS(line)] - tmp) * (1 / sTriangList[(leftListPos)]);
			tmp = 0;
		}
		else
			tmp += vmTmpLocal[POS(column)] * sTriangList[(leftListPos)];
	}

	for (int l = 0 ; l < nComp; l++)
		vmListLocal[POS(l)] = 2 * vmTmpLocal[POS(l)] - vmListLocal[POS(l)];

	//if (h.type == 0 && neuron == 1) printf("vmList=%.4f\n", vmListLocal[POS(0)]);

}

__global__ void solveMatrixG(HinesStruct *hList, int nSteps, int nNeurons, ftype *spikeListGlobal, ftype *weightListGlobal, int *spikeListPosGlobal, int *spikeListStartGlobal, ftype *vmListGlobal) {


	//return;

	//printf("Starting Kernel...\n");

	int neuron = blockIdx.x * blockDim.x + threadIdx.x;
	if (neuron >= nNeurons) return;
	HinesStruct & h = hList[neuron];
	int nComp = h.nComp;
	int triangAll = h.triangAll;
	//int type = h.type;

	if (h.currStep > 0) {
		h.synSpikeListPos = spikeListPosGlobal + neuron * h.synapseListSize;
		h.spikeList = spikeListGlobal;
		h.synapseWeightList = weightListGlobal;
		h.spikeListSize = spikeListStartGlobal[neuron];
	}
	else {
		h.synSpikeListPos = 0;
		h.spikeList = 0;
		h.synapseWeightList = 0;
		h.spikeListSize = 0;
	}

	//	if (neuron==0)
	//		h.curr[0] = 10e-4;// + neuron*1e-4;

	// (ftype * 5 + ucomp * 10) * nComp
	// ftype=4 e ucomp=2 e ncomp = 8   ->  320 bytes 
	// ftype=4 e ucomp=2 e ncomp = 64  -> 2560 bytes  
	extern __shared__ ftype sharedMem[]; 
	ftype *sLeftList       = (ftype *)sharedMem;
	ucomp *sLeftListLine   = (ucomp *)&(sLeftList[h.leftListSize]); 	
	ucomp *sLeftListColumn = (ucomp *)&(sLeftListLine[h.leftListSize]); 

	ftype *sMulList     = (ftype *)&(sLeftListColumn[h.leftListSize]); // mulSize is zero when triangAll is 1
	ucomp *sMulListComp = (ucomp *)&(sMulList[h.mulListSize]); 	
	ucomp *sMulListDest = (ucomp *)&(sMulListComp[h.mulListSize]); 

	ucomp *sLeftStartPos = (ucomp *)&(sMulListDest[h.mulListSize]);


	int nChannelTypes = h.nChannelTypes;
	ftype *sTau		= (ftype *)&(sLeftStartPos[nComp]);
	ftype *sGmax 	= (ftype *)&(sTau[nChannelTypes*2]);
	ftype *sEsyn 	= (ftype *)&(sGmax[nChannelTypes]);

	ftype *lastSharedAddress = (ftype *)&(sEsyn[nChannelTypes]);

	for (int id=0; id < nChannelTypes; id++ ) {
		sTau[2*id]   = h.tau[2*id];
		sTau[2*id+1] = h.tau[2*id+1];
		sGmax[id] 	 = h.gmax[id];
		sEsyn[id] 	 = h.esyn[id];

	}

	for (int k=0; k < nComp; k ++ ) {
		sLeftStartPos[k] = h.leftStartPos[k];
	}

	for (int k=0; k < h.leftListSize; k ++ ) {
		if (triangAll == 0) sLeftList[k] = h.triangList[k];
		else				sLeftList[k] = h.leftList[k];
		sLeftListLine[k]   = h.leftListLine[k];
		sLeftListColumn[k] = h.leftListColumn[k];
	}

	if (triangAll == 0) {
		for (int k=0; k < h.mulListSize; k ++ ) {
			sMulList[k]     = h.mulList[k];
			sMulListComp[k] = h.mulListComp[k];
			sMulListDest[k] = h.mulListDest[k];
		}
	}

	/* 
	 * Allocate for each individual neuron
	 * nThreads * nComp * ftype * 2
	 * 32 * [8 ] * 4 * 2 = 32 * 64 = 2K 	
	 * 32 * [32] * 4 * 2 =         = 8K
	 */
	ftype *rhsLocal = (ftype *)lastSharedAddress;
	ftype *vmListLocal = rhsLocal + blockDim.x * nComp;

	ftype *nGate = vmListLocal + blockDim.x * nComp;
	ftype *hGate = nGate + blockDim.x * h.compListSize;
	ftype *mGate = hGate + blockDim.x * h.compListSize;

	//ftype *spkTimes   = mGate    + blockDim.x * h.compListSize;
	//ftype *spkWeigths = spkTimes + blockDim.x;

	ftype *freeMem = mGate    + blockDim.x * h.compListSize;//spkWeigths + blockDim.x;
	ftype *sTriangList = 0;

	//printf ("used sharedMem1=%ld\n", (long)freeMem-(long)sharedMem);

	if (triangAll == 1) {
		sTriangList = freeMem + threadIdx.x * h.leftListSize;
		freeMem  = freeMem + blockDim.x * h.leftListSize;
	}

	//printf ("used sharedMem2=%ld\n", (long)freeMem-(long)sharedMem);


	//printf ("SolveMatrixG: Ok01\n");

	for (int k=0; k < nComp; k++ )
		vmListLocal[POS(k)] = h.vmList[k];

	for (int k=0; k < h.compListSize; k++ ) {
		nGate[POS(k)] = h.n[k];
		hGate[POS(k)] = h.h[k];
		mGate[POS(k)] = h.m[k];
	}

	//__syncthreads();

	//	printf("%f|%f|%f\n", h.tau[0], h.gmax[0], h.esyn[0]);
	//	assert(false);

	ftype dt = h.dt;
	int currStep = h.currStep;
	ucomp nGeneratedSpikes = 0;

	for(int gStep = 0; gStep < nSteps; gStep++ ) {

		//printf ("SolveMatrixG: Ok1\n");
		if (triangAll == 0) {
			updateRhsG(hList, sMulList, sMulListComp, sMulListDest,
					rhsLocal, vmListLocal, nGate, hGate, mGate, sTau, sGmax, sEsyn, freeMem); // RYC
			backSubstituteG(hList, sLeftList, sLeftListLine, sLeftListColumn, rhsLocal, vmListLocal, freeMem); // RYC
		}
		else {

			upperTriangularizeAll(hList, sTriangList, sLeftList, sLeftListLine, sLeftListColumn,
					sLeftStartPos, rhsLocal, vmListLocal, nGate, hGate, mGate, sTau, sGmax, sEsyn,freeMem);
			backSubstituteG(hList, sTriangList, sLeftListLine, sLeftListColumn, rhsLocal, vmListLocal, freeMem); // RYC
		}
		//printf ("SolveMatrixG: Ok2\n");

		for (int k=0; k<nComp; k++) {
			int index = k * nSteps + gStep;
			h.vmTimeSerie[index] = vmListLocal[POS(k)]; // RYC
		}

		currStep = currStep + 1;


		if (vmListLocal[POS(nComp-1)] >= h.threshold && ((currStep * dt) - h.lastSpike) > h.minSpikeInterval) {

			h.spikeTimes[nGeneratedSpikes] = currStep * dt;
			h.lastSpike = currStep * dt;
			nGeneratedSpikes++;
		}

		h.currStep = currStep;
	}

	//printf("spikes=%d neuron=%d\n", nGeneratedSpikes, neuron);
	h.nGeneratedSpikes[neuron] = nGeneratedSpikes;

	//__syncthreads();

	for (int k=0; k<nComp; k++) {
		h.rhsM[k] = rhsLocal[POS(k)];
		h.vmList[k] = vmListLocal[POS(k)]; // RYC
	}

	for (int k=0; k < h.compListSize; k++ ) {
		h.n[k] = nGate[POS(k)];
		h.h[k] = hGate[POS(k)];
		h.m[k] = mGate[POS(k)];
	}

	vmListGlobal[neuron] = vmListLocal[POS(nComp-1)];

	//h.currStep = h.currStep + nSteps;

}
//} // extern C
