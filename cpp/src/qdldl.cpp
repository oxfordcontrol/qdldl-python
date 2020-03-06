// DEBUG
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "qdldl.hpp"

using namespace qdldl;



Solver::Solver(QDLDL_int n, QDLDL_int * Ap, QDLDL_int *Ai, QDLDL_float * Ax){
	// factor and initialize Solver

	// Dimension
	nx = n;
	nnz = Ap[nx];

	// Elimination tree
	etree  = new QDLDL_int[n];
	Lnz = new QDLDL_int[n];

	// L factors
	Lp = new QDLDL_int[n + 1];

	// D
	D = new QDLDL_float[n];
	Dinv = new QDLDL_float[n];

	// Workspace
	iwork = new QDLDL_int[3 * n];
	bwork = new QDLDL_bool[n];
	fwork = new QDLDL_float[n];

	// Permutation
	P = new QDLDL_int[n];
	Pinv = new QDLDL_int[n];

	// Permutation
	QDLDL_int amd_status = amd_l_order(nx, Ap, Ai, P, NULL, NULL);
	if (amd_status < 0)
		throw std::runtime_error(std::string("Error in AMD computation ") + std::to_string(amd_status));


	// cout << "Ap = [ ";
	// for (int i = 0; i < nx + 1; i++) cout << Ap[i] << " ";
	// cout << "]\n";
	// cout << "Ai = [ ";
	// for (int i = 0; i < nnz; i++) cout << Ai[i] << " ";
	// cout << "]\n";
	// cout << "Ax = [ ";
	// for (int i = 0; i < nnz; i++) cout << Ax[i] << " ";
	// cout << "]\n";
	//

	py::print("Here 1");

	// No permutation
	for (int i = 0; i < nx; i++){
		P[i] = i;
	}

	pinv(P, Pinv, n); // Compute inverse permutation

	// Allocate elements of A permuted
	Aperm_p = new QDLDL_int[n+1];
	Aperm_i = new QDLDL_int[nnz];
	Aperm_x = new QDLDL_float[nnz];
	A2Aperm = new QDLDL_int[n];
	QDLDL_int * work_perm = new QDLDL_int[n]();  // Initialize to 0

	// Permute A
	// symperm(n, Ap, Ai, Ax, Aperm_p, Aperm_i, Aperm_x, Pinv, A2Aperm, work_perm);
	// DEBUG
	std::memcpy(Aperm_p, Ap, (n + 1) * sizeof(QDLDL_int));
	std::memcpy(Aperm_i, Ai, nnz * sizeof(QDLDL_int));
	std::memcpy(Aperm_x, Ax, nnz * sizeof(QDLDL_int));

	py::print("Here 2");

	// Compute elimination tree
    int sum_Lnz = QDLDL_etree(n, Aperm_p, Aperm_i, iwork, Lnz, etree);

	if (sum_Lnz < 0)
		throw std::runtime_error(std::string("Input matrix is not quasi-definite, sum_Lnz = ") + std::to_string(sum_Lnz));

	py::print("Here 3");

	// Allocate factor
	Li = new QDLDL_int[sum_Lnz];
	Lx = new QDLDL_float[sum_Lnz];


	// Compute numeric factorization
    QDLDL_factor(nx, Aperm_p, Aperm_i, Aperm_x,
			     Lp, Li, Lx,
				 D, Dinv, Lnz,
				 etree, bwork, iwork, fwork);

	py::print("Here 4");


    // Delete permutaton workspace
	delete [] work_perm;

	py::print("Here 5");

}




QDLDL_float * Solver::solve(QDLDL_float * b){

	auto * x = new QDLDL_float[nx];
	auto work = new QDLDL_float[nx];

    permute_x(nx, work, b, P);
    QDLDL_solve(nx, Lp, Li, Lx, Dinv, work);
    permutet_x(nx, x, work, P);

	return x;

}



void Solver::update(QDLDL_float * Anew_x){

	// Update matrix
	update_A(nnz, Aperm_x, Anew_x, A2Aperm);

	// Compute numeric factorization
    QDLDL_factor(nx, Aperm_p, Aperm_i, Aperm_x,
			     Lp, Li, Lx,
				 D, Dinv, Lnz,
				 etree, bwork, iwork, fwork);

}

Solver::~Solver(){

	delete [] Lp;
	delete [] Li;
	delete [] Lx;
	delete [] D;
	delete [] Dinv;
	delete [] P;
	delete [] Pinv;
	delete [] etree;
	delete [] Lnz;
	delete [] iwork;
	delete [] bwork;
	delete [] fwork;
	delete [] Aperm_p;
	delete [] Aperm_i;
	delete [] Aperm_x;
	delete [] A2Aperm;

}
