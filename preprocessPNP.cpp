#include "pnpf.h"

using namespace Eigen;
using namespace std;

void create_C_A(Ref<const MatrixXd> P, int tarPtNum, Ref<MatrixXd> A, Ref<MatrixXd> C, Ref<MatrixXd> P_ed)
{
    double P_x_sum = P.row(0).sum();
    double P_y_sum = P.row(1).sum();
    double P_z_sum = P.row(2).sum();

    int n = P.cols();

    if (tarPtNum == 4)
    {
        C.col(0) << P_x_sum/n, P_y_sum/n, P_z_sum/n;
        C.col(1) << P_x_sum/n + 1, P_y_sum/n, P_z_sum/n;
        C.col(2) << P_x_sum/n, P_y_sum/n + 1, P_z_sum/n;
        C.col(3) << P_x_sum/n, P_y_sum/n, P_z_sum/n + 1;

        VectorXd E_n = VectorXd::Constant(n, 1);
        //MatrixXd P_ed(P.rows()+1, P.cols());    
        P_ed.block(0, 0, P.rows(), P.cols()) = P;
        P_ed.row(P_ed.rows()-1) = E_n;

        VectorXd E_4 = VectorXd::Constant(4, 1);
        MatrixXd C_ed(C.rows()+1, C.cols());
        C_ed.block(0, 0, C.rows(), C.cols()) = C;
        C_ed.row(C_ed.rows()-1) = E_4;

        //MatrixXd A(C_ed.rows(), P_ed.cols());
        A = C_ed.colPivHouseholderQr().solve(P_ed); 
    }
    else
    {
        Vector3d C1;
        C1 << P_x_sum/n, P_y_sum/n, P_z_sum/n;

        VectorXd v = VectorXd::Constant(P.cols(), 1);
        MatrixXd Pc = P - C1*v.transpose();
        JacobiSVD<MatrixXd> svd(Pc.transpose(), ComputeFullV);

        MatrixXd V = svd.matrixV();
        VectorXd C2 = C1 + V.col(0);
        VectorXd C3 = C1 + V.col(1);
        C.col(0) = C1;
        C.col(1) = C2;
        C.col(2) = C3;

        VectorXd E_n = VectorXd::Constant(n, 1);
        //MatrixXd P_ed(P.rows()+1, P.cols());    
        P_ed.block(0, 0, P.rows(), P.cols()) = P;
        P_ed.row(P_ed.rows()-1) = E_n;
        VectorXd E_3 = VectorXd::Constant(3, 1);
        MatrixXd C_ed(4, 3);
        C_ed.block<3, 3>(0, 0) = C.block<3, 3>(0, 0);
        C_ed.row(3) = E_3;

        //MatrixXd A(C_ed.rows(), P_ed.cols());
        A = C_ed.colPivHouseholderQr().solve(P_ed); 
    }
}

void up(Ref<const MatrixXd> U, Ref<const MatrixXd> A, Ref<MatrixXd> up_n, int n)
{
    MatrixXd diag = MatrixXd::Zero(U.cols(), U.cols());
    for (int i  = 0; i < U.cols(); i++)
    {
        diag(i, i) = U(n, i);
    }

    up_n = -diag*A.transpose();
}

void preprocessPNP(Ref<const MatrixXd> P, Ref<const MatrixXd> U, int tarPtNum, Ref<MatrixXd> C,
                   Ref<MatrixXd> V, Ref<MatrixXd> D)
{
    MatrixXd P_ed(P.rows()+1, P.cols());  
    MatrixXd A(tarPtNum, P_ed.cols());

    create_C_A(P, tarPtNum, A, C, P_ed);

    MatrixXd up1(U.cols(), A.rows());
    up(U, A, up1, 0);

    MatrixXd up2(U.cols(), A.rows());
    up(U, A, up2, 1);

    //int tarPtNum = 4;
    MatrixXd A2 = A*A.transpose();
    MatrixXd AU1 = A*up1;
    MatrixXd AU2 = A*up2;
    MatrixXd U12 = up1.transpose()*up1 + up2.transpose()*up2;
    MatrixXd M1 = MatrixXd::Zero(2*tarPtNum + U12.rows(), 2*tarPtNum + AU1.cols());
    M1.block(0, 0, tarPtNum, tarPtNum) = A2;
    M1.block(0, 2*tarPtNum, AU1.rows(), AU1.cols()) = AU1;
    M1.block(tarPtNum, tarPtNum, A2.rows(), A2.cols()) = A2;
    M1.block(tarPtNum, 2*tarPtNum, AU2.rows(), AU2.cols()) = AU2;
    M1.block(2*tarPtNum, 0, AU1.cols(), AU1.rows()) = AU1.transpose();
    M1.block(2*tarPtNum, AU1.rows(), AU2.cols(), AU2.rows()) = AU2.transpose();
    M1.block(2*tarPtNum, AU1.rows()+AU2.rows(), U12.rows(), U12.cols()) = U12;
    
    VectorXi inds(12);
    if (tarPtNum == 4)
    {
        inds << 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11;
    }
    else
    {
        inds << 0, 3, 6, 1, 4, 7, 2, 5, 8;
    }
    
    JacobiSVD<MatrixXd> svd(M1, ComputeFullV);
    //MatrixXd D_loc = MatrixXd::Zero(M1.rows(), M1.cols());
    for (int i = 0; i < D.rows(); i++)
    {
        D(i, i) = svd.singularValues()(inds(i));
    }
    MatrixXd V0 = svd.matrixV();
    //MatrixXd V_loc(V0.rows(), V0.cols());

    for(int i = 0; i < 3*tarPtNum; i++)
    {
        V.row(i) = V0.row(inds(i));
    }
}

