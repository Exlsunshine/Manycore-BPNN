#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <cmath>


#define Data  100
#define In 1
#define Out 1
// #define Neuron 45
// #define TrainC 20000
#define Neuron 20
#define TrainC 10000
#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3

double d_in[Data][In], d_out[Data][Out];
double w[Neuron][In], o[Neuron], v[Out][Neuron];
double Maxin[In], Minin[In], Maxout[Out], Minout[Out];
double OutputData[Out];
double dv[Out][Neuron], dw[Neuron][In];
double e;

double getRSSByDistance2(double dist)
{
    //return std::abs(-34 - 10 * 5.0 * log10(dist) + 0.4);
	return (-34 - 10 * 5.0 * log10(dist) + 0.4);
}

void writeTest(){
	FILE *fp1,*fp2;
	double r1,r2,r3;
	int i;
	srand((unsigned)time(NULL)); 
	if((fp1=fopen("in.txt","w"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
	if((fp2=fopen("out.txt","w"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}


	for(i=0;i<Data;i++){
		r2 = 1 + rand() % 34 / 1.0;
		fprintf(fp1,"%lf \n", r2);
		fprintf(fp2,"%lf \n", getRSSByDistance2(r2));
	}
	fclose(fp1);
	fclose(fp2);


    /*



    for(i=0;i<Data;i++)
    {
		for(int j=0; j<In; j++)
			printf("%lf ",d_in[i][j]);

        printf("\n");
    }

	for(i=0;i<Data;i++)
	{
        for(int j=0; j<Out; j++)
		{
            printf("%lf ",d_out[i][j]);
        }
        printf("\n");
    }

    printf("End\n");
    getchar();
*/

}

void readData(){

	FILE *fp1,*fp2;
	int i,j;
	if((fp1=fopen("in.txt","r"))==NULL){
		printf("can not open the in file\n");
		exit(0);
	}
    
    //printf("Reading... \n");
	for(i=0;i<Data;i++)
    {
		for(j=0; j<In; j++)
        {
			fscanf(fp1,"%lf ",&d_in[i][j]);
      //      printf("%lf ", d_in[i][j]);
        }
       // printf("\n");
    }
	fclose(fp1);

	if((fp2=fopen("out.txt","r"))==NULL){
		printf("can not open the out file\n");
		exit(0);
	}
	for(i=0;i<Data;i++)
    {
		for(j=0; j<Out; j++)
        {
			fscanf(fp2,"%lf ",&d_out[i][j]);
      //      printf("%lf ", d_out[i][j]);
        }
    //    printf("\n");
    }
   // printf("Reading...end \n");
	fclose(fp2);
}

void initBPNework()
{
	int i, j;

	for(i = 0; i < In; i++)
	{
		Minin[i] = Maxin[i] = d_in[0][i];
		for(j = 0; j < Data; j++)
		{
			Maxin[i] = Maxin[i] > d_in[j][i] ? Maxin[i] : d_in[j][i];
			Minin[i] = Minin[i] < d_in[j][i] ? Minin[i] : d_in[j][i];
		}
	}

	for(i = 0; i < Out; i++)
	{
		Minout[i] = Maxout[i] = d_out[0][i];
		for(j = 0; j < Data; j++)
		{
			Maxout[i] = Maxout[i] > d_out[j][i] ? Maxout[i] : d_out[j][i];
			Minout[i] = Minout[i] < d_out[j][i] ? Minout[i] : d_out[j][i];
		}
	}

	for (i = 0; i < In; i++)
	{
		for (j = 0; j < Data; j++)
		{
			d_in[j][i] = (d_in[j][i] - Minin[i] + 1) / (Maxin[i] - Minin[i] + 1);
			//printf("int %d is %lf | max = %lf, min = %lf\n", j, d_in[j][i], Maxin[i] , Minin[i]);
		}
	}
	
	//getchar();

	for (i = 0; i < Out; i++)
	{
		for (j = 0; j < Data; j++)
		{
			d_out[j][i] = (d_out[j][i] - Minout[i] + 1) / (Maxout[i] - Minout[i] + 1);
			//printf("out %d is %lf \n", j, d_out[j][i]);
		}
	}

	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < In; ++j){	
			w[i][j] = rand() * 2.0 / RAND_MAX - 1;
			dw[i][j] = 0;
		} 
	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < Out; ++j){
			v[j][i] = rand() * 2.0 / RAND_MAX - 1;
			dv[j][i] = 0;
		}
}

void computO(int var)
{
	int i, j;
	double sum, y;

	for (i = 0; i < Neuron; ++i)
	{
		sum = 0;
		for (j = 0; j < In; ++j)
			sum += w[i][j] * d_in[var][j];
		o[i] = 1 / (1 + exp(-1 * sum));
	}

	for (i = 0; i < Out; ++i)
	{
		sum=0;
		for (j = 0; j < Neuron; ++j)
			sum += v[i][j] * o[j];
		OutputData[i] = sum;
	}	
}

void backUpdate(int var)
{
	int i,j;
	double t;
	for (i = 0; i < Neuron; ++i)
	{
		t=0;
		for (j = 0; j < Out; ++j){
			t+=(OutputData[j]-d_out[var][j])*v[j][i];

			dv[j][i]=A*dv[j][i]+B*(OutputData[j]-d_out[var][j])*o[i];
			v[j][i]-=dv[j][i];
		}

		for (j = 0; j < In; ++j){
			dw[i][j]=a*dw[i][j]+b*t*o[i]*(1-o[i])*d_in[var][j];
			w[i][j]-=dw[i][j];
		}
	}
}

double result(double var1)//, double var2)//, double var3)
{
	int i,j;
	double sum,y;

	var1=(var1-Minin[0]+1)/(Maxin[0]-Minin[0]+1);

	for (i = 0; i < Neuron; ++i){
		sum=0;
		sum=w[i][0]*var1;// + w[i][1]*var2;//+w[i][2]*var3;
		o[i]=1/(1+exp(-1*sum));
	}
	sum=0;
	for (j = 0; j < Neuron; ++j)
		sum+=v[0][j]*o[j];

	return sum*(Maxout[0]-Minout[0]+1)+Minout[0]-1;
}

void writeNeuron()
{
	FILE *fp1;
	int i,j;
	if((fp1=fopen("neuron.txt","w"))==NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < In; ++j){
			fprintf(fp1,"%lf ",w[i][j]);
		}
	fprintf(fp1,"\n\n\n\n");

	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < Out; ++j){
			fprintf(fp1,"%lf ",v[j][i]);
		}

	fclose(fp1);
}

void  trainNetwork()
{
	int i, j, c = 0;
	do
	{
		e = 0;
		for (i = 0; i < Data; ++i)
		{
			computO(i);
			for (j = 0; j < Out; ++j)
            {
		      //  printf("%lf  %lf\n", OutputData[j], d_out[i][j]);
               // getchar();
                e += fabs((OutputData[j] - d_out[i][j]) / d_out[i][j]);
            }
			backUpdate(i);
		}

        double xxxx = Data;
	//	printf("%lf  %lf\n", e , xxxx);
		printf("%d  %lf\n", c, e / Data);
		c++;
	}while(c < TrainC && e / Data > 0.01);
}

int  main(int argc, char const *argv[])
{
	writeTest();
	readData();
/*
    for(int i=0;i<Data;i++)
    {
		for(int j=0; j<In; j++)
			printf("%d : %f = %f ",i, d_in[i][j] , d_out[i][j]);

        printf("\n");
    }
	getchar();*/

	initBPNework();
	trainNetwork();
	printf("%lf = [%lf]\n",result(10) , getRSSByDistance2(60));
	printf("%lf = [%lf]\n",result(21), getRSSByDistance2(21));
	printf("%lf = [%lf]\n",result(4), getRSSByDistance2(4));
	printf("%lf = [%lf]\n",result(24), getRSSByDistance2(24));
	printf("%lf = [%lf]\n",result(18), getRSSByDistance2(18));
	printf("%lf = [%lf]\n",result(9), getRSSByDistance2(9));
	// writeNeuron();
	
	getchar();
    return 0;
}
