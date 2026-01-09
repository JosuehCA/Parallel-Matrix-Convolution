#include <iostream>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include "FreeImage.h"
#include <stdlib.h>
#include "mpi.h"

using namespace std;

FIBITMAP *foto, *fotoconv;
int ancho, alto;
BYTE *pixelOld, *pixelNew;
double startTime, endTime;

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

	FreeImage_Save(FIF_JPEG, foto, "fotogris.jpg", 0);

	fotoconv = FreeImage_Allocate(ancho, alto, 8);

	pixelOld = (BYTE *) FreeImage_GetBits(foto);
	pixelNew = (BYTE *) FreeImage_GetBits(fotoconv);
}

void saveAndStopImageLibraryWork() {
	FreeImage_Save(FIF_JPEG, fotoconv, "fotoconv.jpg",0);
	FreeImage_DeInitialise();
}

int main(int argc, char *argv[]) {
	int id, num_p;
	int i, j, coc, res, tam, myN, h, w, base;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_p);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if(id == 0) {
		startImageLibraryWork();
		startTime = MPI_Wtime();
	}

	MPI_Bcast(&ancho, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&alto, 1, MPI_INT, 0, MPI_COMM_WORLD);

	vector<BYTE> oldbuf(ancho * alto);

	if(id == 0) memcpy(oldbuf.data(), pixelOld, ancho * alto);

	MPI_Bcast(oldbuf.data(), ancho * alto, MPI_BYTE, 0, MPI_COMM_WORLD);

	h = alto - 2;
	w = ancho - 2;
	tam = h * w;

	coc = tam / num_p;
	res = tam % num_p;

	base = id * coc;

	if(id != num_p - 1) myN = coc;
	else myN = coc + res;

	vector<BYTE> loc(myN);

	for(int t = 0; t < myN; t++) {
		int k = base + t;

		i = k / w + 1;
		j = k % w + 1;

		long long sum =
	        kernel[0][0]*oldbuf[(i-1)*ancho + (j-1)] +
	        kernel[0][1]*oldbuf[(i-1)*ancho + (j  )] +
	        kernel[0][2]*oldbuf[(i-1)*ancho + (j+1)] +
	        kernel[1][0]*oldbuf[(i  )*ancho + (j-1)] +
	        kernel[1][1]*oldbuf[(i  )*ancho + (j  )] +
	        kernel[1][2]*oldbuf[(i  )*ancho + (j+1)] +
	        kernel[2][0]*oldbuf[(i+1)*ancho + (j-1)] +
	        kernel[2][1]*oldbuf[(i+1)*ancho + (j  )] +
	        kernel[2][2]*oldbuf[(i+1)*ancho + (j+1)];

		sum = max(0LL, min(255LL, sum));
		loc[t] = sum;
	}

	vector<int> counts, displacements;
	vector<BYTE> glob;

	if(id == 0) {
		counts.resize(num_p);
		displacements.resize(num_p);
		glob.resize(tam);

		for(int r = 0; r < num_p; r++) {
		    counts[r] = (r == num_p - 1) ? (coc + res) : coc;
		    displacements[r] = r * coc;
		}
	}

	// Sería &loc, &glob etc si utilizara arrays estáticos
	MPI_Gatherv(loc.data(), myN, MPI_BYTE, glob.data(), counts.data(), displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);

	if(id == 0) {
		for(int k = 0; k < tam; k++) {
			int i = k / w + 1;
			int j = k % w + 1;

			pixelNew[i * ancho + j] = glob[k];
		}

		endTime = MPI_Wtime();
		cout << "Duración algoritmo paralelo (MPI): " << fixed << setprecision(0) << (endTime - startTime)*1e6 << " microsegundos\n";

		saveAndStopImageLibraryWork();
	}
	MPI_Finalize();
	return 0;
}