#include <iostream>
#include <armadillo>
#include <bitset>
#include <cmath>
#include <iomanip>

using std::cout;
using std::endl;

namespace parameters
{
    constexpr uint sites = 14;        //var1     // no. of sites
    uint states = pow(2, sites);     // no. of states, 2^sites
    constexpr double delta = -0.5;    //var2 // delta in Heisenberg model
    constexpr double t = -0.5;       //var3
    constexpr double epsilon = 1e-8; // degeneracy tolerance
    arma::mat eigenvec;              // matrix of eigenvectors of hamiltonian, eigvec.t().ham.eigvec = diag
    arma::vec eigenval;
} // namespace parameters

namespace operators
{
    uint basic_op = 4;                                         // number of basic operators - here is 4 because we have Id, S^+, S^- and S^z
    uint max_supp = 3;                                         // maximal support of operators
    uint max_op = parameters::sites * pow(basic_op, max_supp); // sites*4^max_supp; 4 because we have Id, S^+, S^-, S^z
    uint no_op;                                                // number of orthogonal operators;
    arma::umat operators_info(max_op, max_supp + 1);
    arma::umat operators_ts(max_op, max_supp + 1);
    /*
        In operators_info there is full information about set of orthogonal operators
        In operators info there is only first op. from set of all translated operators

        operators_info(i,0:max_supp) - info about i-th operator
        oparators_info(i,0) = 0 - do not use this operator
        operators_info(i,0) = 1,..., sites - first site from support
        operators_info(i,j) = operators on site j = 1,...,max_supp from the support
        operators_info(i,j) = 0,1,2,3 (0: 1_j, 1:S^+_j, 2:S^z, 3:S^-_j)
        sum_{j = 1,...,max_supp} must be odd, because we want operators that do not
        conserve number of particles (spin), that is, we want different number
        of S^+ and S^-
        
    */
} // namespace operators

typedef std::bitset<parameters::sites> bits;
arma::wall_clock timer;

int generate_hamiltonian_dense();
int generate_operators();
arma::uvec dec_to_base(uint); // converts decimal number to base specified by number of "basic" operators
arma::mat generate_op_matrix(int);
arma::mat generate_current_matrix(int);

int main()
{

    using namespace parameters;
    using namespace operators;

    timer.tic();
    cout << "#Fermion code\n";
    generate_hamiltonian_dense();
    generate_operators();
    uint cnt = 0;
    for (uint i = 0; i < no_op; i++)
    {
        cout << i + 1 << "   " << operators_info(i, 0) + 1 << " "; // << operators_info(i, 1) << operators_info(i, 2) << operators_info(i, 3) << endl;
        for (uint j = 1; j <= max_supp; j++)
            cout << operators_info(i, j);
        cout << endl;
        if ((i + 1) / sites * sites == i + 1)
        {
            cout << endl;
            cnt++;
        }
    }
    cout << "#" << no_op << " operators generated." << endl;
    cout << "#" << cnt << " operators generated excluding translational symmetry" << endl;
    operators_ts.resize(cnt, max_supp + 1);

    for (uint i = 0; i < cnt; i++)
    {
        operators_ts(i, 0) = operators_info(i * sites, 0);

        // cout << i + 1 << "   " << operators_info(i * sites, 0) + 1 << " "; // << operators_info(i, 1) << operators_info(i, 2) << operators_info(i, 3) << endl;
        for (uint j = 1; j <= max_supp; j++)
        {
            // cout << operators_info(i * sites, j);
            operators_ts(i, j) = operators_info(i * sites, j);
        }
    }

    for (uint i = 0; i < cnt; i++)
    {

        cout << i + 1 << "   " << operators_ts(i, 0) + 1 << " "; // << operators_info(i, 1) << operators_info(i, 2) << operators_info(i, 3) << endl;
        for (uint j = 1; j <= max_supp; j++)
            cout << operators_ts(i, j);
        cout << endl;
    }

    arma::mat corr_mat(cnt, cnt); // Correlation matrix of time-averaged operators op
    corr_mat.zeros();

    for (uint i = 0; i < cnt; i++)
    {
        arma::mat op1 = generate_op_matrix(i);
        for (uint ii = 0; ii < cnt; ii++)
        {
            arma::mat op2 = generate_op_matrix(ii);
            corr_mat(i, ii) = arma::trace(op1.t() * op2) / static_cast<double>(states);
        }
    }
    /*
        Clear eigenval and eigenvec as they are no longer needed,
        corr_mat has been generated at this point.
        Reuse them for eigenvectors and eigenvalues of corr_mat.  
    */
    eigenval.clear();
    eigenvec.clear();
    eigenval.resize(cnt);
    eigenvec.resize(cnt, cnt);
    arma::eig_sym(eigenval, eigenvec, corr_mat);

    cout << endl<<"#Correlation matrix eigenvalues: " << endl;
    for (auto val : eigenval)
        cout << std::setprecision(17) << val << endl;
    cout << endl<<"#10 eigenvectors corresponding to largest eigenvalues: \n";
    cout << std::setprecision(5);
    for (uint i = 0; i < cnt; i++)
    {
        cout << i + 1 << "    ";
        for (int j = 1; j <= 10; j++)
        {
            arma::vec vec = eigenvec.col(cnt - j);
            cout << vec(i) << "    ";
        }
        cout << "\n";
    }
    cout << "#Execution time: " << timer.toc() << " s" << endl;
    return 0;
}

int generate_hamiltonian_dense()
{

    using namespace parameters;
    arma::mat hamiltonian(states, states);
    hamiltonian.zeros();
    for (uint i = 0; i < states; i++)
    {
        bits state(i);
        uint no_fermions = state.count();

        for (uint ii = 0; ii < sites; ii++)
        {
            double sign = 1.0;
            int first = ii;
            int second = ii + 1;
            if (second == sites)
            {
                if (no_fermions % 2 == 0)
                    sign = -1.0;
                second = 0;
            }

            bool s1 = state.test(first);
            bool s2 = state.test(second);
            hamiltonian(i, i) += delta * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s2) - 0.5);

            if (s1 == false && s2 == true)
            {
                state[first] = true;
                state[second] = false;
                hamiltonian(i, state.to_ulong()) += -t * sign;
                state[first] = false;
                state[second] = true;
            }
            if (s1 == true && s2 == false)
            {
                state[first] = false;
                state[second] = true;
                hamiltonian(i, state.to_ulong()) += -t * sign;
                state[first] = true;
                state[second] = false;
            }
        }
    }

    cout << "#Is hermitian: " << hamiltonian.is_symmetric() << endl;
    arma::eig_sym(eigenval, eigenvec, hamiltonian);

    return 0;
}

int generate_operators()
{
    using namespace parameters;
    using namespace operators;
    operators_info.zeros();
    uint last_op = 0; // number of last saved operator

    for (uint i = 0; i < pow(basic_op, max_supp); i++)
    {

        arma::uvec op_temp = dec_to_base(i);

        if (op_temp(0) == 0)
        {
            continue;
        } // we don't want Id on first site

        int parity = 0;
        for (uint ii = 0; ii < max_supp; ii++) // we want different number of S^+ and S^-
        {
            if (op_temp(ii) == 1)
                parity++;
            if (op_temp(ii) == 3)
                parity--;
        }
        if (parity == 0)
            continue;

        arma::uvec op_temp_hc = op_temp;
        int op_temp_hc_index = 0;
        for (uint ii = 0; ii < max_supp; ii++) // building hermitian conjugate of current operator
        {
            if (op_temp_hc(ii) == 1)
                op_temp_hc(ii) = 3;
            else if (op_temp_hc(ii) == 3)
                op_temp_hc(ii) = 1;

            op_temp_hc_index += op_temp_hc(ii) * pow(basic_op, ii);
        }

        bool bad_op_flag = false;
        for (uint ii = 0; ii < last_op; ii++) // check if this operator is a hermitian conjugate of already saved one
        {
            int op_ii_index = 0;
            for (uint iii = 1; iii < max_supp + 1; iii++)
                op_ii_index += operators_info(ii, iii) * pow(basic_op, iii - 1);

            if (op_ii_index == op_temp_hc_index)
            {
                bad_op_flag = true;
                break;
            }
        }

        if (bad_op_flag)
        {
            continue;
        }

        // operators arriving past this point pass all test and are saved to operators_info

        for (uint ii = 0; ii < sites; ii++)
        {
            operators_info(last_op, 0) = ii;

            for (uint iii = 1; iii < max_supp + 1; iii++)
                operators_info(last_op, iii) = op_temp(iii - 1);

            last_op++;
        }
    }
    operators_info.resize(last_op, max_supp + 1);
    no_op = last_op;
    return 0;
}

arma::uvec dec_to_base(uint num)
{
    arma::uvec temp(operators::max_supp);

    for (uint i = 0; i < operators::max_supp; i++)
    {
        temp[i] = num % operators::basic_op;
        num /= operators::basic_op;
    }

    return temp;
}

arma::mat generate_current_matrix(int index)
{
    /*
        This function generates matrix of operator using operators_info(index,...)
        and eigenvectors of the hamiltonian to represent them in basis of
        hamiltonian eigenstates.     
    */
    using namespace parameters;
    using namespace operators;
    arma::mat op(states, states); // initialize matrix of the operator
    op.zeros();

    for (uint i = 0; i < states; i++)
    {
        //bits initial_state(i);
        bits final_state(i);
        uint first_site_index = operators_info(index, 0);
        double mat_el = 1.0; // matrix element

        for (uint ii = 0; ii < max_supp; ii++)
        {
            uint site_index = first_site_index + ii;
            if (site_index >= sites)
                site_index -= sites;

            switch (operators_info(index, ii + 1))
            {
            case 0: // Identity operator
            {
                // Do nothing
                break;
            }
            case 1: // S^+_{site_index}
            {
                double ap;
                if (final_state.test(site_index))
                {
                    ap = 0.0;
                    //do not change final state
                }
                else
                {

                    ap = 1;
                    if (site_index > 0)
                    {
                        for (uint iii = 0; iii < site_index; iii++) // +1 or -1 phase from Jordan-Wigner transformation
                        {                                           // from spin operators to fermionic operators
                            if (final_state.test(iii))
                                ap *= (-1);
                        }
                    }
                    final_state[site_index] = true; // Modify final state
                }
                mat_el *= ap;
                break;
            }
            case 2: // S^z_{site_index}
            {
                double sz = 0;
                if (final_state.test(site_index))
                    sz = 1;
                mat_el *= (sz - 0.5) * sqrt(2.0);
                // Do not change final state
                break;
            }
            case 3: // S^-_{site_index}
            {
                double am;
                if (final_state.test(site_index))
                {

                    am = 1;
                    if (site_index > 0)
                    {
                        for (uint iii = 0; iii < site_index; iii++) // +1 or -1 phase from Jordan-Wigner transformation
                        {                                           // from spin operators to fermionic operators
                            if (final_state.test(iii))
                                am *= (-1);
                        }
                    }
                    final_state[site_index] = false; // Modify final state
                }
                else
                {
                    am = 0.0;
                    // do not modify final state
                }
                mat_el *= am;
                break;
            }
            default:
                cout << "#There is no such operator.\n Someting went wrong.\n";
                break;
            }
        } // end of support loop

        op(i, final_state.to_ulong()) += mat_el;
    } // end of basis states loop

    /*
    substrac Hermitian conjugate - actually just transpose because this matrix is real
    actually we want current operator - i*(A-A^T), but the factor i is irrelevant for
    Hilbert-Schmidt product 
    */
    op -= arma::trans(op);

    // transform op matrix to basis of Hamiltonian eigenstates op --> eigvec^T.op.eigvec
    op = arma::trans(eigenvec) * op * eigenvec;

    // Traceless
    double trace = arma::trace(op) / states;
    op.diag() -= trace;

    // Normed
    double norm = sqrt(arma::trace(op.t() * op) / states);
    op /= norm;

    // Time averaging
    for (uint j = 0; j < states; j++)
    {
        for (uint jj = 0; jj < states; jj++)
        {
            if (std::abs(eigenval(j) - eigenval(jj)) > epsilon)
            {
                op(j, jj) = 0.0;
            }
        }
    }

    return op;
}

arma::mat generate_op_matrix(int index)
{
    /*
        This function generates matrix of operator using operators_info(index,...)
        and eigenvectors of the hamiltonian to represent them in basis of
        hamiltonian eigenstates.     
        
        Takes into account translational symmetry of operators
    
    */
    using namespace parameters;
    using namespace operators;
    double mult = 1.0;

    arma::mat op(states, states); // initialize matrix of the operator
    op.zeros();
    for (uint shift = 0; shift < sites; shift++)
    {  
        for (uint i = 0; i < states; i++)
        {
            //bits initial_state(i);
            bits final_state(i);
            uint first_site_index = operators_ts(index, 0) + shift;
            double mat_el = 1.0; // matrix element

            for (uint ii = 0; ii < max_supp; ii++)
            {
                uint site_index = first_site_index + ii;
                if (site_index >= sites)
                    site_index -= sites;

                switch (operators_ts(index, ii + 1))
                {
                case 0: // Identity operator
                {
                    // Do nothing
                    break;
                }
                case 1: // S^+_{site_index}
                {
                    double ap;
                    if (final_state.test(site_index))
                    {
                        ap = 0.0;
                        //do not change final state
                    }
                    else
                    {

                        ap = 1;
                        if (site_index > 0)
                        {
                            for (uint iii = 0; iii < site_index; iii++) // +1 or -1 phase from Jordan-Wigner transformation
                            {                                           // from spin operators to fermionic operators
                                if (final_state.test(iii))
                                    ap *= (-1);
                            }
                        }
                        final_state[site_index] = true; // Modify final state
                    }
                    mat_el *= ap;
                    break;
                }
                case 2: // S^z_{site_index}
                {
                    double sz = 0;
                    if (final_state.test(site_index))
                        sz = 1;
                    mat_el *= (sz - 0.5) * sqrt(2.0);
                    // Do not change final state
                    break;
                }
                case 3: // S^-_{site_index}
                {
                    double am;
                    if (final_state.test(site_index))
                    {

                        am = 1;
                        if (site_index > 0)
                        {
                            for (uint iii = 0; iii < site_index; iii++) // +1 or -1 phase from Jordan-Wigner transformation
                            {                                           // from spin operators to fermionic operators
                                if (final_state.test(iii))
                                    am *= (-1);
                            }
                        }
                        final_state[site_index] = false; // Modify final state
                    }
                    else
                    {
                        am = 0.0;
                        // do not modify final state
                    }
                    mat_el *= am;
                    break;
                }
                default:
                    cout << "#There is no such operator.\n Someting went wrong.\n";
                    break;
                }
            } // end of support loop

            op(i, final_state.to_ulong()) += mat_el * mult;
        } // end of basis states loop
        if (delta < 0.0 && (sites == 10 || sites == 14)) mult*=-1.0;
    }
    // add Hermitian conjugate - actually just transpose because this matrix is real
    op += arma::trans(op);

    // transform op matrix to basis of Hamiltonian eigenstates op --> eigvec^T.op.eigvec
    op = arma::trans(eigenvec) * op * eigenvec;

    // Traceless
    double trace = arma::trace(op) / states;
    op.diag() -= trace;

    // Normed
    double norm = sqrt(arma::trace(op.t() * op) / states);
    op /= norm;

    // Time averaging
    for (uint j = 0; j < states; j++)
    {
        for (uint jj = 0; jj < states; jj++)
        {
            if (std::abs(eigenval(j) - eigenval(jj)) > epsilon)
            {
                op(j, jj) = 0.0;
            }
        }
    }

    return op;
}
