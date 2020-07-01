/*
 * nn.c
 *
 *  Created on: 5 jul. 2016
 *  Author: ecesar
 *
 *      Descripció:
 *      Xarxa neuronal simple de tres capes. La d'entrada que són els pixels d'una
 *      imatge (mirar descripció del format al comentari de readImg) de 32x32 (un total de 1024
 *      entrades). La capa oculta amb un nombre variable de neurones (amb l'exemple proporcionat 117
 *      funciona relativament bé, però si incrementem el nombre de patrons d'entrament caldrà variar-lo).
 *      Finalment, la capa de sortida (que ara té 10 neurones ja que l'entrenem per reconéixer 10
 *      patrons ['0'..'9']).
 *      El programa passa per una fase d'entrenament en la qual processa un conjunt de patrons (en
 *      l'exemple proporcionat són 1934 amb els dígits '0'..'9', escrits a mà). Un cop ha calculat 
 * 	    els pesos entre la capa d'entrada i l'oculta i entre
 *      aquesta i la de sortida, passa a la fase de reconèixament, on llegeix 946 patrons d'entrada
 *      (es proporcionen exemples per aquests patrons), i intenta reconèixer de quin dígit es tracta.
 *
 *  Darrera modificació: gener 2019. Ara l'aprenentatge fa servir la tècnica dels mini-batches
 */

/*******************************************************************************
*    Aquest programa és una adaptació del fet per  JOHN BULLINARIA  
*    ( http://www.cs.bham.ac.uk/~jxb/NN/nn.html):
*
*    nn.c   1.0                                       � JOHN BULLINARIA  2004  *
*******************************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <limits.h>
#include "common.h"
#include <omp.h>
int total;
int seed=50;

inline int rando()
{
    seed = (214013*seed+2531011);
    return seed>>16;
}

inline double frando()
{
    return (rando()/65536.0f);
}

void freeDeltaWeights(double *DeltaWeightIH[], double *DeltaWeightHO[]){
	for( int i = 0; i < NUMHID; i++)
		free(DeltaWeightIH[i]);
	for( int i = 0; i < NUMOUT; i++)
		free(DeltaWeightHO[i]);
}

void freeWeights(double *WeightIH[],  double *WeightHO[]){
	for( int i = 0; i < NUMHID; i++)
		free(WeightIH[i]);
	for( int i = 0; i < NUMOUT; i++)
		free(WeightHO[i]);
}

void freeTSet( int np, char **tset ){
	for( int i = 0; i < np; i++ ) free( tset[i] );
	free(tset);
}

void trainN(){

	char **tSet;

	if( (tSet = loadPatternSet(NUMPAT, "optdigits.tra", 1)) == NULL){
        printf("Loading Patterns: Error!!\n");
		exit(-1);
	}

    	double *DeltaWeightIH[NUMHID], smallwt = 0.22;

	for( int i = 0; i < NUMHID; i++){
		if ((WeightIH[i] = (double *)malloc((NUMIN)*sizeof(double))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}
		if ((DeltaWeightIH[i] = (double *)malloc((NUMIN)*sizeof(double))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}
	}
	for(int j = 0; j < NUMIN; j++)
		for( int i = 0; i < NUMHID; i++){
			WeightIH[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			DeltaWeightIH[i][j] = 0.0;
		}
	
	double *DeltaWeightHO[NUMOUT];

	for( int i = 0; i < NUMOUT; i++){
		if ((WeightHO[i] = (double *)malloc((NUMHID)*sizeof(double))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}
		if ((DeltaWeightHO[i] = (double *)malloc((NUMHID)*sizeof(double))) == NULL){
			printf("Out of Mem\n");
			exit(-1);
		}
	}
	
	for(int j = 0; j < NUMHID; j++)
		for( int i = 0; i < NUMOUT; i++){
			WeightHO[i][j] = 2.0 * ( frando() + 0.01 ) * smallwt;
			DeltaWeightHO[i][j] = 0.0;
		}
	

    double Error, BError, eta = 0.3, alpha = 0.5;
	int ranpat[NUMPAT];
	double Hidden[NUMPAT][NUMHID], Output[NUMPAT][NUMOUT], DeltaO[NUMOUT], DeltaH[NUMHID];
		#pragma omp parallel
		{		
				
			for( int epoch = 0 ; epoch < 1000000 ; epoch++) {    // iterate weight updates
			
				#pragma omp for
				for( int p = 0 ; p < NUMPAT ; p++ )   // randomize order of individuals
					{ranpat[p] = p ;}	
				
				#pragma omp single
				{
					for( int p = 0 ; p < NUMPAT ; p++) {
						int x = rando();
						int np = (x*x)%NUMPAT;
						int op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
					}
				
					Error = BError = 0.0;
				
					printf("."); fflush(stdout);
				}
				
				for (int nb = 0; nb < NUMPAT/BSIZE; nb++) { // repeat for all batches
					BError = 0.0;
					
					for( int np = nb*BSIZE ; np < (nb + 1)*BSIZE ; np++ ) {    // repeat for all the training patterns within the batch
						int p = ranpat[np];
							
							//for 1
							#pragma omp for
							for( int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
								double SumH = 0.0;
								for( int i = 0 ; i < NUMIN ; i++ ) SumH += tSet[p][i] * WeightIH[j][i]; //*** UTILITZAR PATRON REDUCE
								Hidden[p][j] = 1.0/(1.0 + exp(-SumH)) ;
							}
							
							//for 2
							#pragma omp for reduction(+:BError)
							for( int k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations and errors
								double SumO = 0.0;
								for( int j = 0 ; j < NUMHID ; j++ ) SumO += Hidden[p][j] * WeightHO[k][j] ; //*** UTILITZAR PATRON REDUCE
								Output[p][k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
								//*** UTILITZAR PATRON REDUCE
								BError += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   // SSE
								DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   // Sigmoidal Outputs, SSE
							}
							
							//for 3
							#pragma omp for
							for( int j = 0 ; j < NUMHID ; j++ ) {     // update delta weights DeltaWeightIH
								double SumDOW = 0.0 ;
								for( int k = 0 ; k < NUMOUT ; k++ ) SumDOW += WeightHO[k][j] * DeltaO[k] ; //*** UTILITZAR PATRON REDUCE
								DeltaH[j] = SumDOW * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
							}
							
							//new for
							#pragma omp for collapse(2)
							for( int j = 0 ; j < NUMHID ; j++ ) {     // update delta weights DeltaWeightIH
								for( int i = 0 ; i < NUMIN ; i++ )
									DeltaWeightIH[j][i] = eta * tSet[p][i] * DeltaH[j] + alpha * DeltaWeightIH[j][i];
							}
							
							//int k,j;
							//for 4 (UTILIZAR COLLAPSE)
							#pragma omp for collapse(2)
							for(int k = 0 ; k < NUMOUT ; k ++ )    // update delta weights DeltaWeightHO
								for(int j = 0 ; j < NUMHID ; j++ )
									DeltaWeightHO[k][j] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[k][j];
						
					}
					
					#pragma omp single	
					{
						Error += BError;
					}
					
					//int i,j,k;
					#pragma omp for collapse(2)
					for(int j = 0 ; j < NUMHID ; j++ )     // update weights WeightIH
						for(int i = 0 ; i < NUMIN ; i++ )
							WeightIH[j][i] += DeltaWeightIH[j][i] ;
							
					#pragma omp for collapse(2)
					for(int k = 0 ; k < NUMOUT ; k ++ )    // update weights WeightHO
						for(int j = 0 ; j < NUMHID ; j++ ) 
							WeightHO[k][j] += DeltaWeightHO[k][j] ;
					
				}
				
				#pragma omp single
				{
					Error = Error/((NUMPAT/BSIZE)*BSIZE);	//mean error for the last epoch 		
					if( !(epoch%100) ) printf("\nEpoch %-5d :   Error = %f \n", epoch, Error) ;
				}
				
				if( Error < 0.0004 ) {
						printf("\nEpoch %-5d :   Error = %f \n", epoch, Error) ; break ;  // stop learning when 'near enough'
				}
			#pragma omp barrier
    	}
	}
	freeDeltaWeights(DeltaWeightIH, DeltaWeightHO);
	freeTSet( NUMPAT, tSet );

	for( int p = 0 ; p < NUMPAT ; p++ ) {
		printf("\n%d\t", p) ;
		for( int k = 0 ; k < NUMOUT ; k++ ) {
				printf("%f\t%f\t", Target[p][k], Output[p][k]) ;
		}
	}
	printf("\n"); 
}

void printRecognized(int p, double Output[]){
	int imax = 0;

	for( int i = 1; i < NUMOUT; i++)
		if ( Output[i] > Output[imax] ) imax = i;
	printf("El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p]);
	if ( imax == Validation[p] ) total++;
    for( int k = 0 ; k < NUMOUT ; k++ )
        	printf("\t%f\t", Output[k]) ;
    printf("\n");
}

void runN(){
	char **rSet;
	char *fname[NUMRPAT];

	if( (rSet = loadPatternSet(NUMRPAT, "optdigits.cv", 0)) == NULL){
		printf("Error!!\n");
		exit(-1);
	}

	double Hidden[NUMHID], Output[NUMOUT];
		#pragma omp parallel
		{
			for( int p = 0 ; p < NUMRPAT ; p++ ) {    // repeat for all the recognition patterns
				#pragma omp for
				for( int j = 0 ; j < NUMHID ; j++ ) {    // compute hidden unit activations
						double SumH = 0.0;
						for( int i = 0 ; i < NUMIN ; i++ ) SumH += rSet[p][i] * WeightIH[j][i]; //*** UTILITZAR PATRON REDUCE
						Hidden[j] = 1.0/(1.0 + exp(-SumH)) ;
				}
				#pragma omp for
				for( int k = 0 ; k < NUMOUT ; k++ ) {    // compute output unit activations
						double SumO = 0.0;
						for( int j = 0 ; j < NUMHID ; j++ ) SumO += Hidden[j] * WeightHO[k][j] ; //*** UTILITZAR PATRON REDUCE
						Output[k] = 1.0/(1.0 + exp(-SumO)) ;   // Sigmoidal Outputs
				}
				#pragma omp single
				{
					printRecognized(p, Output);
				}
			}
		}

	printf("\nTotal encerts = %d\n", total);

	freeTSet( NUMRPAT, rSet );
}

int main() {
	clock_t start = clock();
	srand(start); 		//Comentat porta a resultats deterministes (però en el cas real ha d'aparéixer)
	trainN();
	runN();

	freeWeights(WeightIH, WeightHO);
	clock_t end = clock();
	printf("\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC)) ;

    	return 1 ;
}

/*******************************************************************************/


