g++ -c -O0 -g3 SynapticChannels.cpp launchHines.cpp ActiveChannels.cpp HinesMatrix.cpp PlatformFunctions.cpp SynapticChannels.cpp Connections.cpp SpikeStatistics.cpp
nvcc $1 -O0 -G -maxrregcount=96 --ptxas-options=-v -arch sm_13 -o HinesGpu HinesGpu.cu PrepareGpuExecution.cu SynapticComm.cu launchHines.o ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o
# --use_fast_math
#-gencode arch=compute_13,code=sm_13
#-arch sm_13b
