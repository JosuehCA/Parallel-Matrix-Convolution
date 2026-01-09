#include <iostream>
#include "FreeImage.h"
#include <stdlib.h>
#include <cuda_runtime.h>

#define THR_POR_BLK 512

using namespace std;

FIBITMAP *foto, *fotoconv;
int ancho, alto, dimension;
BYTE *pixelOld, *pixelNew;

__constant__ int d_kernel[9];

void startImageLibraryWork() {
	FreeImage_Initialise();

	foto = FreeImage_Load(FIF_JPEG, "foto.jpg", JPEG_GREYSCALE);
	ancho = FreeImage_GetWidth(foto);
	alto = FreeImage_GetHeight(foto);
	dimension = ancho * alto;

	FreeImage_Save(FIF_JPEG, foto, "fotogris.jpg", 0);

	fotoconv = FreeImage_Allocate(ancho, alto, 8);

	pixelOld = (BYTE *) FreeImage_GetBits(foto);
	pixelNew = (BYTE *) FreeImage_GetBits(fotoconv);
}

void saveAndStopImageLibraryWork() {
	FreeImage_Save(FIF_JPEG, fotoconv, "fotoconv.jpg",0);
	FreeImage_DeInitialise();
}

__global__ void worker(BYTE *pixelOld, BYTE *pixelNew, int ancho, int alto, int tam) {
	int w = ancho - 2;

	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= tam) return;

	int i = 1 + k / w;
    int j = 1 + k % w;

	long long sum =
	  d_kernel[0]*pixelOld[(i-1) * ancho + (j-1)]
	+ d_kernel[1]*pixelOld[(i-1) * ancho + (j)]
	+ d_kernel[2]*pixelOld[(i-1) * ancho + (j+1)]
	+ d_kernel[3]*pixelOld[(i) 	 * ancho + (j-1)]
	+ d_kernel[4]*pixelOld[(i) 	 * ancho + (j)]
	+ d_kernel[5]*pixelOld[(i) 	 * ancho + (j+1)]
	+ d_kernel[6]*pixelOld[(i+1) * ancho + (j-1)]
	+ d_kernel[7]*pixelOld[(i+1) * ancho + (j)]
	+ d_kernel[8]*pixelOld[(i+1) * ancho + (j+1)];

	pixelNew[i * ancho + j] = sum;
}

int main() {
	startImageLibraryWork();

	int h_kernel[9] = {		// Embossing
		-2, -1, 0,
		-1, 1, 1,
		0, 1, 2
	};

	int bytes = ancho * alto * sizeof(BYTE);

	cudaEvent_t ini, fin;
    cudaEventCreate(&ini);
    cudaEventCreate(&fin);
    cudaEventRecord(ini, 0);

  	BYTE *d_old, *d_new;

  	cudaMalloc(&d_old, bytes);
  	cudaMalloc(&d_new, bytes);
  	cudaMemcpy(d_old, pixelOld, bytes, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_new, pixelNew, bytes, cudaMemcpyHostToDevice);
  	cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel));

  	int w = ancho - 2;
  	int h = alto - 2;
  	int tam = w * h;
  	int blocks = (tam + THR_POR_BLK - 1) / THR_POR_BLK;

  	worker<<< blocks, THR_POR_BLK >>>(d_old, d_new, ancho, alto, tam);

  	cudaMemcpy(pixelNew, d_new, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(fin, 0);
    cudaEventSynchronize(fin);

    float tiempo;
    cudaEventElapsedTime(&tiempo, ini, fin);
	cout << "Duración algoritmo paralelo (CUDA con " << THR_POR_BLK << " threads por bloque): " << tiempo*1000 << " microsegundos\n";

	saveAndStopImageLibraryWork();

	cudaEventDestroy(ini);
    cudaEventDestroy(fin);
    cudaFree(d_old);
    cudaFree(d_new);
}
