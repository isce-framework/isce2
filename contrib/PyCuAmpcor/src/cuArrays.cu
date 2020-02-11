
#include "cuArrays.h"
#include "cudaError.h"
	
	template <typename T>
	void cuArrays<T>::allocate()
	{
		checkCudaErrors(cudaMalloc((void **)&devData, getByteSize()));
        is_allocated = 1; 
	}
	
    template <typename T>
    void cuArrays<T>::allocateHost()
    {
        hostData = (T *)malloc(getByteSize());
        //checkCudaErrors(cudaMallocHost((void **)&hostData, getByteSize()));
        is_allocatedHost = 1;
    }
    
	template <typename T>
	void cuArrays<T>::deallocate()
	{
		checkCudaErrors(cudaFree(devData));
        is_allocated = 0; 
	}
	
    template <typename T>
	void cuArrays<T>::deallocateHost()
	{
		//checkCudaErrors(cudaFreeHost(hostData));
        free(hostData);
        is_allocatedHost = 0; 
	}
    
    template <typename T>
	void cuArrays<T>::copyToHost(cudaStream_t stream)
	{
        //std::cout << "debug copy " << is_allocatedHost << " " << is_allocated  << " " << getByteSize() << "\n";
		checkCudaErrors(cudaMemcpyAsync(hostData, devData, getByteSize(), cudaMemcpyDeviceToHost, stream));
	}
    
    template <typename T>
    void cuArrays<T>::copyToDevice(cudaStream_t stream)
	{
		checkCudaErrors(cudaMemcpyAsync(devData, hostData, getByteSize(), cudaMemcpyHostToDevice, stream));
	}
    
    template <typename T>
    void cuArrays<T>::setZero(cudaStream_t stream)
    {
        checkCudaErrors(cudaMemsetAsync(devData, 0, getByteSize(), stream));
    }
    
	template<>
	void cuArrays<float2>::debuginfo(cudaStream_t stream) {
		//std::cout << height << " " << width << " " << count << std::endl;
        //std::cout << height << " " << width << " " << count << std::endl;
        if( !is_allocatedHost)
    		allocateHost();
        copyToHost(stream);
    
        //cudaStreamSynchronize(stream);
        //std::cout << "debug debuginfo " << size << " " << count << " " << stream << "\n";

		int range = min(10, size*count);
	
		for(int i=0; i<range; i++)
			std::cout << "(" <<hostData[i].x << " ," << hostData[i].y << ")" ;
		std::cout << std::endl;
        if(size*count>range) {
            for(int i=size*count-range; i<size*count; i++)
                std::cout << "(" <<hostData[i].x << " ," << hostData[i].y << ")" ;
            std::cout << std::endl;
        }
	}
	
    	
	template<>
	void cuArrays<int2>::debuginfo(cudaStream_t stream) {
		//std::cout << height << " " << width << " " << count << std::endl;
        if( !is_allocatedHost)
    		allocateHost();
        copyToHost(stream);
		int range = min(10, size*count);
	
		for(int i=0; i<range; i++)
			std::cout << "(" <<hostData[i].x << " ," << hostData[i].y << ")" ;
		std::cout << std::endl;
		if(size*count>range) {
            for(int i=size*count-range; i<size*count; i++)
                std::cout << "(" <<hostData[i].x << " ," << hostData[i].y << ")" ;
            std::cout << std::endl;
        }
	}
    
	template <>
	void cuArrays<float>::debuginfo(cudaStream_t stream) {
		std::cout << height << " " << width << " " << count << std::endl;
        if( !is_allocatedHost)
    		allocateHost();
        copyToHost(stream);
		
		int range = min(10, size*count);
	
		for(int i=0; i<range; i++)
			std::cout << "(" <<hostData[i]  << ")" ;
		std::cout << std::endl;
		if(size*count>range) {
            for(int i=size*count-range; i<size*count; i++)
                std::cout << "(" <<hostData[i] << ")" ;
            std::cout << std::endl;
        }
	}
	
	template<typename T>
	void cuArrays<T>::outputToFile(std::string fn, cudaStream_t stream)
	{
        if( !is_allocatedHost)
    		allocateHost();
        copyToHost(stream);
        outputHostToFile(fn);
	}

    template <typename T>
    void cuArrays<T>::outputHostToFile(std::string fn)
	{
		std::ofstream file;  
		file.open(fn.c_str(),  std::ios_base::binary);
		file.write((char *)hostData, getByteSize());
		file.close();
	}
    
	/*
	template<>
	void cuArrays<float>::outputToFile(std::string fn, cudaStream_t stream)
	{
		float *data;
		data = (float *)malloc(size*count*sizeof(float));
		cudaMemcpyAsync(data, devData, size*count*sizeof(float), cudaMemcpyDeviceToHost, stream);
		std::ofstream file;  
		file.open(fn.c_str(),  std::ios_base::binary);
		file.write((char *)data, size*count*sizeof(float));
		file.close();
	}*/
	
	template<>
	void cuArrays<float2>::outputToFile(std::string fn, cudaStream_t stream)
	{
		float *data;
		data = (float *)malloc(size*count*sizeof(float2));
		checkCudaErrors(cudaMemcpyAsync(data, devData, size*count*sizeof(float2), cudaMemcpyDeviceToHost, stream));
		std::ofstream file;  
		file.open(fn.c_str(),  std::ios_base::binary);
		file.write((char *)data, size*count*sizeof(float2));
		file.close();
	}

	template<>
	void cuArrays<float3>::outputToFile(std::string fn, cudaStream_t stream)
	{
		float *data;
		data = (float *)malloc(size*count*sizeof(float3));
		checkCudaErrors(cudaMemcpyAsync(data, devData, size*count*sizeof(float3), cudaMemcpyDeviceToHost, stream));
		std::ofstream file;  
		file.open(fn.c_str(),  std::ios_base::binary);
		file.write((char *)data, size*count*sizeof(float3));
		file.close();
	}
	
	template class cuArrays<float>;
	template class cuArrays<float2>;
    template class cuArrays<float3>;
	template class cuArrays<int2>;
    template class cuArrays<int>;
