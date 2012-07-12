MPI_COMPILE_FLAGS="-DMPI_GPU_NN -I/usr/include/mpich2 -I/home/rcamargo/cuda/include/"
#$(echo $(mpicc --showme:compile))
MPI_LINK_FLAGS="-L/usr/lib -lmpich -lopa -lmpl -lrt -lcr -lpthread" 
#$(echo $(mpicc --showme:link))
        
g++ -c -O3  ${MPI_COMPILE_FLAGS} SynapticChannels.cpp launchHines.cpp ActiveChannels.cpp HinesMatrix.cpp PlatformFunctions.cpp SynapticChannels.cpp Connections.cpp SpikeStatistics.cpp NeuronInfoWriter.cpp PerformSimulation.cpp CpuSimulationControl.cpp

nvcc $1 ${MPI_COMPILE_FLAGS} ${MPI_LINK_FLAGS} --compiler-options -O3 --ptxas-options=-v -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20 -o HinesGpu ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o launchHines.o NeuronInfoWriter.o PerformSimulation.o CpuSimulationControl.o HinesGpu.cu GpuSimulationControl.cu SynapticComm.cu

#nvcc $1 ${MPI_COMPILE_FLAGS} ${MPI_LINK_FLAGS} --compiler-options -O3 --ptxas-options=-v -m64 -arch sm_30 -o HinesGpu ActiveChannels.o HinesMatrix.o PlatformFunctions.o SynapticChannels.o Connections.o SpikeStatistics.o launchHines.o NeuronInfoWriter.o PerformSimulation.o CpuSimulationControl.o HinesGpu.cu GpuSimulationControl.cu SynapticComm.cu


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
