#ifndef PNPF_H
#define PNPF_H

#include <iostream>
#include <Eigen>
#include <vector>
#include <cmath>
#include "ClpSimplex.hpp"
#include "CoinPackedMatrix.hpp"
#include "CoinPackedVector.hpp"
#include "levmar.h"
#define FROM_MATR_TO_RODR 0
#define FROM_RODR_TO_MATR 1

using namespace Eigen;
using namespace std;

class pnpf
{
public:
    double f;
    Matrix3d R;
    Vector3d t;
    double avgErr;
};

class pnpOpts
{
public:
    double fMax;
    double fMin;
    double errThr;
    int isFastBA;
    //haveMex
};

void pnpfmy(Ref<const MatrixXd> pts, Ref<const MatrixXd> Uc, int tarPtNum, 
                   int isFast, class pnpOpts &Opts, double &f, MatrixXd &R1, VectorXd &t1);

void preprocessPNP(Ref<const MatrixXd> P, Ref<const MatrixXd> U, int tarPtNum, Ref<MatrixXd> C,
                   Ref<MatrixXd> V, Ref<MatrixXd> D);

template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = numeric_limits<double>::epsilon());

void de2bi(int i, int n, Ref<VectorXd> binVec);

void alphaFormula(Ref<const VectorXd> x1, Ref<const VectorXd> x2, VectorXd &vMy, double &alphaMy);

void rodrigues(MatrixXd &R, double *V, int type);

void pnpNd1204Multi(Ref<const MatrixXd> C, Ref<const MatrixXd> V, Ref<const MatrixXd> D, int tarPtNum,
	Ref<const VectorXd> Cf, int fans, int N, int stopN, Ref<const MatrixXd> pts, int isFast, MatrixXd &rest, VectorXd &test, double &fest, double &retVal);

void pnp1(Ref<const MatrixXd> C, Ref<const MatrixXd> V, Ref<const MatrixXd> D, int tarPtNum,
	Ref<const VectorXd> Cf, int fans, Ref<const MatrixXd> pts, int isFast, MatrixXd &rest, VectorXd &test, double &fest, double &retVal);

void adjustInLKer(Ref<const VectorXd> x0, Ref<const MatrixXd> kerVect, int betaNum, Ref<const MatrixXd> L, int isFast, VectorXd &vMy, double &alphaMy, int &errSign);

void adjustInLKerMulti(Ref<const VectorXd> x0, Ref<const MatrixXd> kerVect, int betaNum, Ref<const MatrixXd> L, int isFast, MatrixXd &xkNs);

void collectResult(double retVal, MatrixXd &Rest, VectorXd &test, double &fest, Ref<const MatrixXd> Uc, Ref<const MatrixXd> pts, class pnpf &currentSolution, int isFast, const class pnpOpts &pnpOpts);

#endif