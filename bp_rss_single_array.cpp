#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <cmath>


#define Data  710
#define In 1
#define Out 1
#define Neuron 20
#define TrainC 10000
#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3

double d_in[Data], d_out[Data];
double w[Neuron], o[Neuron], v[Neuron];
double Maxin, Minin, Maxout, Minout;
double OutputData;
double dv[Neuron], dw[Neuron];
double e;

void readData() {

	FILE *fp1, *fp2;
	int i, j;
	if ((fp1 = fopen("in.txt", "r")) == NULL) {
		printf("can not open the in file\n");
		exit(0);
	}

	for (i = 0; i < Data; i++)
		fscanf(fp1, "%lf ", &d_in[i]);
	fclose(fp1);

	if ((fp2 = fopen("out.txt", "r")) == NULL) {
		printf("can not open the out file\n");
		exit(0);
	}
	for (i = 0; i < Data; i++)
		fscanf(fp2, "%lf ", &d_out[i]);
	fclose(fp2);
}

void initBPNework()
{
	int i, j;

	Minin = Maxin = d_in[0];
	for (j = 0; j < Data; j++)
	{
		Maxin = Maxin > d_in[j] ? Maxin : d_in[j];
		Minin = Minin < d_in[j] ? Minin : d_in[j];
	}

	Minout = Maxout = d_out[0];
	for (j = 0; j < Data; j++)
	{
		Maxout = Maxout > d_out[j] ? Maxout : d_out[j];
		Minout = Minout < d_out[j] ? Minout : d_out[j];
	}

	for (j = 0; j < Data; j++)
		d_in[j] = (d_in[j] - Minin + 1) / (Maxin - Minin + 1);

	for (j = 0; j < Data; j++)
		d_out[j] = (d_out[j] - Minout + 1) / (Maxout - Minout + 1);

	for (i = 0; i < Neuron; ++i)
	{
		w[i] = rand() * 2.0 / RAND_MAX - 1;
		dw[i] = 0;
	}
	for (i = 0; i < Neuron; ++i)
	{
		v[i] = rand() * 2.0 / RAND_MAX - 1;
		dv[i] = 0;
	}
}

void computO(int var)
{
	int i, j;
	double sum, y;

	for (i = 0; i < Neuron; ++i)
	{
		sum = 0;
		sum += w[i] * d_in[var];
		o[i] = 1 / (1 + exp(-1 * sum));
	}

	sum = 0;
	for (j = 0; j < Neuron; ++j)
		sum += v[j] * o[j];
	OutputData = sum;
}

void backUpdate(int var)
{
	int i, j;
	double t;
	for (i = 0; i < Neuron; ++i)
	{
		t = 0;
		t += (OutputData - d_out[var])*v[i];

		dv[i] = A*dv[i] + B*(OutputData - d_out[var])*o[i];
		v[i] -= dv[i];

		dw[i] = a*dw[i] + b*t*o[i] * (1 - o[i])*d_in[var];
		w[i] -= dw[i];
	}
}

void writeNeuron()
{
	FILE *fp1;
	int i, j;
	if ((fp1 = fopen("neuron.txt", "w")) == NULL)
	{
		printf("can not open the neuron file\n");
		exit(0);
	}
	for (i = 0; i < Neuron; ++i)
		fprintf(fp1, "%lf ", w[i]);
	fprintf(fp1, "\n\n\n\n");

	for (i = 0; i < Neuron; ++i)
		fprintf(fp1, "%lf ", v[i]);

	fclose(fp1);
}

void trainNetwork()
{
	int i, j, c = 0;
	do
	{
		e = 0;
		for (i = 0; i < Data; ++i)
		{
			computO(i);
			e += fabs((OutputData - d_out[i]) / d_out[i]);
			backUpdate(i);
		}

		c++;
		printf("%d  %lf\n", c, e / Data);
	} while (c < TrainC && e / Data > 0.01);
}

double result(double var1)
{
	int i, j;
	double sum, y;

	var1 = (var1 - Minin + 1) / (Maxin - Minin + 1);

	for (i = 0; i < Neuron; ++i) {
		sum = 0;
		sum = w[i] * var1;
		o[i] = 1 / (1 + exp(-1 * sum));
	}
	sum = 0;
	for (j = 0; j < Neuron; ++j)
		sum += v[j] * o[j];

	return sum*(Maxout - Minout + 1) + Minout - 1;
}

void testNetworkPerformanceByInputAndOutput()
{
	double x[Data], y[Data];
	FILE *fp1, *fp2;

	int i, j;
	if ((fp1 = fopen("in.txt", "r")) == NULL) {
		printf("can not open the in file\n");
		exit(0);
	}
	for (i = 0; i < Data; i++)
		fscanf(fp1, "%lf ", &x[i]);
	fclose(fp1);

	if ((fp2 = fopen("out.txt", "r")) == NULL) {
		printf("can not open the out file\n");
		exit(0);
	}

	for (i = 0; i < Data; i++)
		fscanf(fp2, "%lf ", &y[i]);
	fclose(fp2);

	printf("%lf = [%lf]\n", result(x[1]), (y[1]));
	printf("%lf = [%lf]\n", result(x[10]), (y[10]));
	printf("%lf = [%lf]\n", result(x[100]), (y[100]));
	printf("%lf = [%lf]\n", result(x[500]), (y[500]));
	printf("%lf = [%lf]\n", result(x[600]), (y[600]));
	printf("%lf = [%lf]\n", result(x[700]), (y[700]));
}

int  main(int argc, char const *argv[])
{
	readData();
	initBPNework();
	trainNetwork();
	writeNeuron();

	testNetworkPerformanceByInputAndOutput();
	getchar();
	return 0;
}
