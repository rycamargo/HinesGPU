g++ -I/home/rcamargo/cuda/include/ -c -g SynapticChannels.cpp launchHines.cpp ActiveChannels.cpp HinesMatrix.cpp PlatformFunctions.cpp SynapticChannels.cpp Connections.cpp SpikeStatistics.cpp NeuronInfoWriter.cpp PerformSimulation.cpp CpuSimulationControl.cpp
nvcc $1 -g -G --ptxas-options=-v -m64 -arch sm_30 -o HinesGpu ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o launchHines.o NeuronInfoWriter.o PerformSimulation.o CpuSimulationControl.o HinesGpu.cu GpuSimulationControl.cu SynapticComm.cu

#OpenMPI 1.2
#g++ -c -D_REENTRANT -O3 SynapticChannels.cpp launchHines.cpp ActiveChannels.cpp HinesMatrix.cpp PlatformFunctions.cpp SynapticChannels.cpp Connections.cpp SpikeStatistics.cpp
#nvcc $1 -D_REENTRANT -lmpi_cxx -lmpi -lopen-rte -lopen-pal -lutil --compiler-options -O3 -maxrregcount=124 --ptxas-options=-v -m32 -arch sm_11 -o HinesGpu HinesGpu.cu PrepareGpuExecution.cu SynapticComm.cu launchHines.o ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o



#nvcc $1 -D_REENTRANT -I/usr//include --compiler-options -O3 -maxrregcount=124 --ptxas-options=-v -arch sm_13 -o HinesGpu HinesGpu.cu PrepareGpuExecution.cu SynapticComm.cu launchHines.o ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o
#g++ -std=c++0x -c -O3 SynapticChannels.cpp launchHines.cpp ActiveChannels.cpp HinesMatrix.cpp PlatformFunctions.cpp SynapticChannels.cpp Connections.cpp SpikeStatistics.cpp
#nvcc $1 --compiler-options -std=c++0x -O3 -maxrregcount=128 --ptxas-options=-v -arch sm_11 -o HinesGpu HinesGpu.cu PrepareGpuExecution.cu launchHines.o ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o

# --use_fast_math
#-gencode arch=compute_13,code=sm_13
#-arch sm_13b
# -Xcompiler -std=c++0x
