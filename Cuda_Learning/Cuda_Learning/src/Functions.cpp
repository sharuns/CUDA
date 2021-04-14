#include <stdio.h>
#include "Functions.h"
#include "Enable_Que.h"

void init(int* ptr, int Size) {


	for (int i = 0; i < Size; i++) {
	
		ptr[i] = i + 1;
	}
	
}


void CheckValid(int* first, int* second, int  Size) {

	for (int i = 0; i < Size; i++) {

		if (first[i] != second[i]) {
			printf("Arrays not equal\n");
			return;
		}
	}
	printf("Arrays are equal\n");
}


void SumArray(int* first, int* second, int* res, int size) {
	

	for (int i = 0; i < size; i++) {

		res[i] = first[i] + second[i];
	}
	

}


