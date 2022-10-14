/*
Author: Jibreel Natsheh
Student #: 181039
Course: Parallel Computing
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 3
#define R_Matrix N
#define C_Matrix N
#define R_Vector N

#define MSTag 1
#define SMTag 5

int rank , size ;
int i, j, k;

double a[R_Matrix][C_Matrix];
double b[R_Vector][1];
double mat_result[R_Matrix][1];

double start_time , end_time;
double start_Stime , end_Stime;
double start_Ptime , end_Ptime;
double time_analysis, speedup , efficiency;

int low_bound;
int upper_bound;
int portion;
MPI_Status status;
MPI_Request request;

void fillVectors()
{
  srand(time(NULL));
  for (i = 0; i < R_Matrix; i++) {
      for (j = 0; j < C_Matrix; j++) {
          a[i][j] = (rand() % 10001) / 10000.0; 
      }
  }
  for (i = 0; i < R_Vector; i++) {
      for (j = 0; j < 1; j++) {
          b[i][j] = (rand() % 10001) / 10000.0; 
      }
  }
}
void printArray()
{
    printf("Matrix:\n");
    for (i = 0; i < R_Matrix; i++) {
      for (j = 0; j < C_Matrix; j++)
        printf("%8.2f  ", a[i][j]);
    }
    printf("\n=============================\nVector:\n");
    for (i = 0; i < R_Vector; i++) {
      for (j = 0; j < 1; j++)
        printf("%8.2f", b[i][j]);
    }
    printf("\n=============================\nResult:\n");
    for (i = 0; i < R_Matrix; i++) {
      for (j = 0; j < 1; j++)
        printf("%8.2f  ", mat_result[i][j]);
    }
    printf("\n\n");
    double running_stime, running_ptime;
    running_stime = end_Stime - start_Stime;
    running_ptime = end_Ptime - start_Ptime;
    printf("\nSerial Running Time = %f\n\n", running_stime);
    printf("\nParallel Running Time = %f\n\n", running_ptime);
}
void SerialCalculation() {
   int i, j, k;                                       
    for (i = 0; i < R_Matrix; i++)                 
      for (j = 0; j < 1; j++)      
        for (k = 0; k < R_Vector; k++)
            mat_result[i][j] += (a[i][k] * b[k][j]);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  fillVectors();
  start_Stime = MPI_Wtime();
	SerialCalculation();
	end_Stime = MPI_Wtime();

  if (rank == 0) {
    start_Ptime = MPI_Wtime();
    for (i = 1; i < size; i++) {                               
      portion = (R_Matrix / (size - 1));                                 
      low_bound = (i - 1) * portion;
      if (((i + 1) == size) && ((R_Matrix % (size - 1)) != 0)) 
          upper_bound = R_Matrix;                                          
      else 
          upper_bound = low_bound + portion;                                       
      MPI_Send(&low_bound, 1, MPI_INT, i, MSTag, MPI_COMM_WORLD); 
      MPI_Send(&upper_bound, 1, MPI_INT, i, MSTag + 1, MPI_COMM_WORLD);
      MPI_Send(&a[low_bound][0], (upper_bound - low_bound) * C_Matrix, MPI_DOUBLE, i, MSTag + 2, MPI_COMM_WORLD);
    }
    for (i = 1; i < size; i++) {
      MPI_Recv(&low_bound, 1, MPI_INT, i, SMTag, MPI_COMM_WORLD, &status);
      MPI_Recv(&upper_bound, 1, MPI_INT, i, SMTag + 1, MPI_COMM_WORLD, &status);
      MPI_Recv(&mat_result[low_bound][0], (upper_bound - low_bound) * 1, MPI_DOUBLE, i, SMTag + 2, MPI_COMM_WORLD, &status);
    }
  }
  MPI_Bcast(&b, R_Vector*1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank > 0) 
    {
        MPI_Recv(&low_bound, 1, MPI_INT, 0, MSTag, MPI_COMM_WORLD, &status);   
        MPI_Recv(&upper_bound, 1, MPI_INT, 0, MSTag + 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&a[low_bound][0], (upper_bound - low_bound) * C_Matrix, MPI_DOUBLE, 0, MSTag + 2, MPI_COMM_WORLD, &status);
        for (i = low_bound; i < upper_bound; i++) {
            for (j = 0; j < 1; j++) {
                for (k = 0; k < R_Vector; k++) {
                    mat_result[i][j] += (a[i][k] * b[k][j]);
                }
            }
        }
        MPI_Send(&low_bound, 1, MPI_INT, 0, SMTag, MPI_COMM_WORLD);
        MPI_Send(&upper_bound, 1, MPI_INT, 0, SMTag + 1, MPI_COMM_WORLD);
        MPI_Send(&mat_result[low_bound][0], (upper_bound - low_bound) * 1, MPI_DOUBLE, 0, SMTag + 2, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        for (i = 1; i < size; i++) {
            MPI_Recv(&low_bound, 1, MPI_INT, i, SMTag, MPI_COMM_WORLD, &status);
            MPI_Recv(&upper_bound, 1, MPI_INT, i, SMTag + 1, MPI_COMM_WORLD, &status);
            MPI_Recv(&mat_result[low_bound][0], (upper_bound - low_bound) * 1, MPI_DOUBLE, i, SMTag + 2, MPI_COMM_WORLD, &status);
        }
        end_Ptime = MPI_Wtime();
        printArray();
    }
  MPI_Finalize(); 
  return 0;   
}