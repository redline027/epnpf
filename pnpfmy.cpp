#include "pnpf.h"

using namespace Eigen;
using namespace std;

/* didn't test: begin */
int checkPnPSolution(class pnpf &currentSolution, class pnpOpts &Opts)
{
	int ret = 0;

	if (currentSolution.avgErr < 0)
		return ret;

	if (currentSolution.avgErr > Opts.errThr)
		return ret;

	if (currentSolution.f > Opts.fMax)
		return ret;

	ret = 1;

	return ret;
}


/* didn't test: end */

void pnpfmy(Ref<const MatrixXd> pts, Ref<const MatrixXd> Uc, int tarPtNum, 
                   int isFast, class pnpOpts &Opts, double &f1, MatrixXd &R1, VectorXd &t1)
{
    VectorXd Cf = VectorXd::Zero(tarPtNum * 3);
    int fans = 1;
    int stopN = 3;
    if (isFast || tarPtNum == 3)
    {
        stopN = 2;
    }

    MatrixXd C(3,4);      // ?????? (4x3 - wrong)
    MatrixXd V(3*tarPtNum, 3*tarPtNum); 
    MatrixXd D = MatrixXd::Zero(3*tarPtNum, 3*tarPtNum); 
    
    preprocessPNP(pts, Uc, tarPtNum, C, V, D);

    int N = 1;

    class pnpf currentSolution;
    currentSolution.avgErr = -1;
	currentSolution.f = -1;

    /* didn't test: begin */
    while (!checkPnPSolution(currentSolution, Opts) && N <= stopN)
    {
		MatrixXd rest(0, 0);
		VectorXd test(0);
		double fest;
		double retVal;
		if (N == 1)
        {
			pnp1(C, V, D, tarPtNum, Cf, fans, pts, isFast, rest, test, fest, retVal);
        }
        else
        {
			pnpNd1204Multi(C, V, D, tarPtNum, Cf, fans, N, stopN, pts, isFast, rest, test, fest, retVal);
        }
		
		collectResult(retVal, rest, test, fest, Uc, pts, currentSolution, isFast, Opts);
		
		N++;
		if (N == stopN + 1)
			if (currentSolution.avgErr < 0)
				stopN = 5;
    }

	if ((currentSolution.avgErr < 0 || currentSolution.t.norm() > 100) && isFast == 1)
		pnpfmy(pts, Uc, tarPtNum, 0, Opts, f1, R1, t1);
	else
	{
		f1 = currentSolution.f;
		R1 = currentSolution.R;
		t1 = currentSolution.t;
	}

	/* didn't test: end */
}