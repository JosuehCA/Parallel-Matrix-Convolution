#include <iostream>
#include <chrono>
#include "FreeImage.h"
#include <stdlib.h>
#include <omp.h>

using namespace std;

// imgomp convol_omp && ./convol_omp (alias mío)

FIBITMAP *foto, *fotoconv;
int ancho, alto, dimension;
BYTE *pixelOld, *pixelNew;

int kernel[3][3] = {		// Embossing
	{-2, -1, 0},
	{-1,  1, 1},
	{0,	1, 2}
};

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

int main() {
	startImageLibraryWork();

	int i, j;
	auto init = chrono::steady_clock::now();

	for (i=1; i<alto-1; i++) {		
		for (j=1; j<ancho-1; j++) {
			long long sum = 0;
			sum =
			  kernel[0][0]*pixelOld[(i-1) * ancho + (j-1)]
			+ kernel[0][1]*pixelOld[(i-1) * ancho + (j)]
			+ kernel[0][2]*pixelOld[(i-1) * ancho + (j+1)]
			+ kernel[1][0]*pixelOld[(i) 	* ancho + (j-1)]
			+ kernel[1][1]*pixelOld[(i) 	* ancho + (j)]
			+ kernel[1][2]*pixelOld[(i) 	* ancho + (j+1)]
			+ kernel[2][0]*pixelOld[(i+1) * ancho + (j-1)]
			+ kernel[2][1]*pixelOld[(i+1) * ancho + (j)]
			+ kernel[2][2]*pixelOld[(i+1) * ancho + (j+1)];

			pixelNew[i * ancho + j] = sum;
		}
	}

	auto fin = chrono::steady_clock::now();
	auto tiempo = chrono::duration_cast<chrono::microseconds>(fin - init).count();
	cout << "Duración algoritmo secuencial: " << tiempo << " microsegundos\n";

	for(int threads = 2; threads <= 8; threads+=2) {
		init = chrono::steady_clock::now();
		#pragma omp parallel num_threads(threads)
		{
			#pragma omp for collapse(2) schedule(static)
			for (i=1; i<alto-1; i++) {
				for (j=1; j<ancho-1; j++) {
					long long sum = 0;
					sum =
					  kernel[0][0]*pixelOld[(i-1) * ancho + (j-1)]
					+ kernel[0][1]*pixelOld[(i-1) * ancho + (j)]
					+ kernel[0][2]*pixelOld[(i-1) * ancho + (j+1)]
					+ kernel[1][0]*pixelOld[(i) 	* ancho + (j-1)]
					+ kernel[1][1]*pixelOld[(i) 	* ancho + (j)]
					+ kernel[1][2]*pixelOld[(i) 	* ancho + (j+1)]
					+ kernel[2][0]*pixelOld[(i+1) * ancho + (j-1)]
					+ kernel[2][1]*pixelOld[(i+1) * ancho + (j)]
					+ kernel[2][2]*pixelOld[(i+1) * ancho + (j+1)];

					pixelNew[i * ancho + j] = sum;
				}
			}
		}
		fin = chrono::steady_clock::now();
		tiempo = chrono::duration_cast<chrono::microseconds>(fin - init).count();
		cout << "Duración algoritmo paralelo con " << threads << " threads: " << tiempo << " microsegundos\n";
	}

	saveAndStopImageLibraryWork();
}
