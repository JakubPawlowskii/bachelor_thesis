#include "../Includes/xxz_real.hpp"
#include "../Includes/Timer.h"
#include <armadillo>
#include <iomanip>
#include <omp.h>

struct operators_data
{

    uint sites;
    uint basic_op;             // number of basic operators - here is 4 because we have Id, S^+, S^- and S^z
    uint max_supp;             // maximal support of operators
    uint max_op;               // sites*4^max_supp; 4 because we have Id, S^+, S^-, S^z
    uint no_op;                // number of orthogonal operators;
    double epsilon;            // degeneracy tolerance
    bool flag;                 // true - O+O^dag, false - i(O-O^dag)
    bool commute;              // true - operators commute with total Sz / total particle number, false - operators do not commute
    bool trans_sym;            // true - operators are assumed to be translationally symmetric during contruction, false - no assumptions
    bool flip;                 // true - sign is flipped between shifted operators when taking translationall symmetry into account
    arma::umat operators_info; //(max_op, max_supp + 1);
    /*
        In operators_info there is full information about set of orthogonal operators

        operators_info(i,0:max_supp) - info about i-th operator
        oparators_info(i,0) = 0 - do not use this operator
        operators_info(i,0) = 1,..., sites - first site from support
        operators_info(i,j) = operators on site j = 1,...,max_supp from the support
        operators_info(i,j) = 0,1,2,3 (0: 1_j, 1:S^+_j, 2:S^z, 3:S^-_j)
        sum_{j = 1,...,max_supp} must be odd, because we want operators that do not
        conserve number of particles (spin), that is, we want different number
        of S^+ and S^-
        
    */
};

arma::uvec dec_to_base(uint, uint, uint); // converts decimal number to base specified by number of "basic" operators
int generate_operators(operators_data *);
arma::mat generate_op_matrix(uint, operators_data *, xxz_real *, double);
arma::mat generate_op_matrix_from_vector(arma::vec, operators_data *, xxz_real *, double);
void usage(bool, char *);
void save_op_for_ham(uint, operators_data *, xxz_real *, arma::vec *, arma::mat *);
int main(int argc, char **argv)
{
    omp_set_num_threads(8);
    Timer time;
    //------------------------------- Hamiltonian parameters ---------------------------------------------------
    double t = -0.5;
    double delta = 1.0;
    double alpha = 0.0;
    //----------------------------------------------------------------------------------------------------------

    //------------------------------- Operators parameters -----------------------------------------------------
    operators_data ops;
    ops.sites = 8;
    ops.basic_op = 4;
    ops.max_supp = 3;
    ops.epsilon = 1e-8;
    ops.flag = true;
    ops.commute = false;
    ops.trans_sym = false;
    ops.flip = false;

    uint save_op = 0;
    // Handle command-line option switches
    char *progname = argv[0];
    if (argc == 1)
    {
        std::cout << "# Running default parameters" << std::endl;
    }
    else
    {
        while (1)
        {
            --argc;
            ++argv;
            if ((argc == 0) || (argv[0][0] != '-'))
                break;
            if ((argv[0][1] == '\0') || (argv[0][2] != '\0'))
                usage(false, progname);
            switch (argv[0][1])
            {
            case 'h':
                usage(true, progname);
                exit(0);
                break;
            case 't':
                sscanf(argv[1], "%lf", &t);
                ++argv;
                --argc;
                break;
            case 'd':
                sscanf(argv[1], "%lf", &delta);
                ++argv;
                --argc;
                break;
            case 'a':
                sscanf(argv[1], "%lf", &alpha);
                ++argv;
                --argc;
                break;
            case 's':
                sscanf(argv[1], "%d", &ops.sites);
                ++argv;
                --argc;
                break;
            case 'm':
                sscanf(argv[1], "%d", &ops.max_supp);
                ++argv;
                --argc;
                break;
            case 'S':
                sscanf(argv[1], "%d", &save_op);
                ++argv;
                --argc;
                break;
            case 'G':
                switch (argv[1][0])
                {
                case '0':
                    ops.flag = false;
                    break;
                case '1':
                    ops.flag = true;
                    break;
                default:
                    printf("No such option for -G\n");
                    usage(false, progname);
                    exit(2);
                    break;
                }
                ++argv;
                --argc;
                break;
            case 'F':
                switch (argv[1][0])
                {
                case '0':
                    ops.flip = false;
                    break;
                case '1':
                    ops.flip = true;
                    break;
                default:
                    printf("No such option for -F\n");
                    usage(false, progname);
                    exit(2);
                    break;
                }
                ++argv;
                --argc;
                break;
            case 'C':
                switch (argv[1][0])
                {
                case '0':
                    ops.commute = false;
                    break;
                case '1':
                    ops.commute = true;
                    break;
                default:
                    printf("No such option for -C\n");
                    usage(false, progname);
                    exit(2);
                    break;
                }
                ++argv;
                --argc;
                break;
            case 'T':
                switch (argv[1][0])
                {
                case '0':
                    ops.trans_sym = false;
                    break;
                case '1':
                    ops.trans_sym = true;
                    break;
                default:
                    printf("No such option for -T\n");
                    usage(false, progname);
                    exit(2);
                    break;
                }
                ++argv;
                --argc;
                break;
            default:
                usage(false, progname);
            }
        }
    }

    ops.max_op = ops.sites * pow(ops.basic_op, ops.max_supp);
    ops.no_op = 0;
    ops.operators_info.zeros(ops.max_op, ops.max_supp + 1);
    //----------------------------------------------------------------------------------------------------------

    std::cout << "# Spin code\n\n";
    xxz_real ham(ops.sites, ops.sites/2, t, delta, alpha);
    ham.print_hamiltonian();

    // std::cout << "# Parameters:\n"
    //           << "# Hamiltonian: " << ham.get_name() << "\n# t = " + std::to_string(t) << "\n# delta = " << std::to_string(delta) << "\n# alpha = " << std::to_string(alpha) << "\n";
    // std::cout << "# sites = " << std::to_string(ops.sites) << "\n# support = " << std::to_string(ops.max_supp);

    // if (ops.flag)
    //     std::cout << "\n# O+O^+"
    //               << "\n";
    // else
    //     std::cout << "\n# i(O-O^+)"
    //               << "\n";
    // std::cout << "# commutes = " << std::to_string(ops.commute) << "\n# translational symmetry = " << std::to_string(ops.trans_sym) << "\n# flip = " << std::to_string(ops.flip) << std::endl;

    // ham.diagonalize();

    // if (ham.is_diag())
    //     std::cout << "#Hamiltonian diagonalized.\n";
    // else
    // {
    //     std::cerr << "Could not diagonalize Hamiltonian. Aborting..." << std::endl;
    //     exit(EXIT_FAILURE);
    // }

    // generate_operators(&ops);

    // if (ops.trans_sym)
    // {
    //     ops.no_op /= ops.sites;
    //     for (uint i = 1; i < ops.no_op; i++)
    //     {
    //         ops.operators_info(i, 0) = ops.operators_info(i * ops.sites, 0);
    //         for (uint j = 1; j <= ops.max_supp; j++)
    //         {
    //             ops.operators_info(i, j) = ops.operators_info(i * ops.sites, j);
    //         }
    //     }
    //     ops.operators_info.resize(ops.no_op, ops.max_supp + 1);
    // }

    // for (uint i = 0; i < ops.no_op; i++)
    // {
    //     std::cout << i + 1 << "   " << ops.operators_info(i, 0) + 1 << " ";
    //     for (uint j = 1; j <= ops.max_supp; j++)
    //         std::cout << ops.operators_info(i, j);
    //     std::cout << std::endl;
    //     if (!ops.trans_sym && (i + 1) / ops.sites * ops.sites == i + 1)
    //         std::cout << std::endl;
    // }
    // std::cout << "#" << ops.no_op << " operators generated." << std::endl;

    // arma::mat corr_mat(ops.no_op, ops.no_op); // Correlation matrix of time-averaged operators op
    // corr_mat.zeros();

    // for (uint i = 0; i < ops.no_op; i++)
    // {
    //     arma::mat op1 = generate_op_matrix(i, &ops, &ham, ops.epsilon);
    //     for (uint ii = 0; ii < ops.no_op; ii++)
    //     {
    //         arma::mat op2 = generate_op_matrix(ii, &ops, &ham, ops.epsilon);
    //         corr_mat(i, ii) = arma::trace(op1.t() * op2) / static_cast<double>(ham.get_n_states());
    //     }
    // }

    // /*
    //     Clear eigenval and eigenvec as they are no longer needed,
    //     corr_mat has been generated at this point.
    // */

    // ham.clear_ham_matrix();
    // if (save_op == 0)
    // {
    //     std::cout << "#Cleared eigenvalues and eigenvectors.\n";
    //     ham.clear_eigenvalues();
    //     ham.clear_eigenvectors();
    // }

    // arma::vec eigenvalues(ops.no_op);
    // arma::mat eigenvectors(ops.no_op, ops.no_op);

    // arma::eig_sym(eigenvalues, eigenvectors, corr_mat);

    // eigenvectors.clean(1e-15);
    
    
    // std::cout << "#Correlation matrix eigenvalues: " << std::endl;
    // for (uint i = 0; i < ops.no_op; i++)
    //     std::cout << std::setprecision(15) << eigenvalues(i) << std::endl;
    // std::cout << "#5 eigenvectors corresponding to largest eigenvalues: \n";
    // for (uint i = 0; i < ops.no_op; i++)
    // {
    //     if (!ops.trans_sym && i != 0 && i % ops.sites == 0)
    //         std::cout << std::endl;
    //     std::cout << std::setw(3) << i + 1;
    //     for (int j = 1; j <= 5; j++)
    //     {
    //         arma::vec vec = eigenvectors.col(ops.no_op - j);
    //         std::cout << std::setprecision(10) << std::setw(17) << vec(i) << "    ";
    //     }
    //     std::cout << "\n";
    // }

    // if (save_op)
    // {

    //     // save_op_for_ham(save_op, &ops, &ham, &eigenvalues, &eigenvectors);
    //     std::vector<double> alphas = {0.0, 0.05,0.1,0.2};
    //     xxz_real* ham2;
    //     for(double alpha : alphas)
    //     {
    //         ham2 = new xxz_real(ops.sites, t, delta, alpha);
    //         save_op_for_ham(save_op, &ops, ham2, &eigenvalues, &eigenvectors);
    //         delete ham2;            
    //     }
    //     ham2 = nullptr;
    // }

    // std::cout << "Executed in " << std::setprecision(4) << time.elapsed() << " seconds" << std::endl;

    return 0;
}

int generate_operators(operators_data *ops)
{
    uint last_op = 0; // number of last saved operator

    for (uint i = 0; i < pow(ops->basic_op, ops->max_supp); i++)
    {

        arma::uvec op_temp = dec_to_base(i, ops->max_supp, ops->basic_op);
        // std::cout<<op_temp<<std::endl;

        if (op_temp(0) == 0)
        {
            continue;
        } // we don't want Id on first site

        int parity = 0;
        for (uint ii = 0; ii < ops->max_supp; ii++) // we want different number of S^+ and S^-
        {
            if (op_temp(ii) == 1)
                parity++;
            if (op_temp(ii) == 3)
                parity--;
        }

        if (ops->commute)
        {
            if (parity != 0)
                continue;
        }
        else
        {
            if (parity == 0)
                continue;
        }

        if (!ops->flag) // we don't want operators made only from S^z if we consider currents
        {
            int other_ops_count = 0;
            for (uint ii = 0; ii < ops->max_supp; ii++)
            {
                if (op_temp(ii) == 1)
                    other_ops_count++;
                if (op_temp(ii) == 3)
                    other_ops_count++;
            }
            if (other_ops_count == 0)
                continue;
        }

        arma::uvec op_temp_hc = op_temp;
        int op_temp_hc_index = 0;
        for (uint ii = 0; ii < ops->max_supp; ii++) // building hermitian conjugate of current operator
        {
            if (op_temp_hc(ii) == 1)
                op_temp_hc(ii) = 3;
            else if (op_temp_hc(ii) == 3)
                op_temp_hc(ii) = 1;

            op_temp_hc_index += op_temp_hc(ii) * pow(ops->basic_op, ii);
        }

        bool bad_op_flag = false;
        for (uint ii = 0; ii < last_op; ii++) // check if this operator is a hermitian conjugate of already saved one
        {
            int op_ii_index = 0;
            for (uint iii = 1; iii < ops->max_supp + 1; iii++)
                op_ii_index += ops->operators_info(ii, iii) * pow(ops->basic_op, iii - 1);

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

        for (uint ii = 0; ii < ops->sites; ii++)
        {
            ops->operators_info(last_op, 0) = ii;

            for (uint iii = 1; iii < ops->max_supp + 1; iii++)
                ops->operators_info(last_op, iii) = op_temp(iii - 1);

            last_op++;
        }
    }
    ops->operators_info.resize(last_op, ops->max_supp + 1);
    ops->no_op = last_op;
    return 0;
}

arma::uvec dec_to_base(uint num, uint bits, uint base)
{
    arma::uvec temp(bits);

    for (uint i = 0; i < bits; i++)
    {
        temp[i] = num % base;
        num /= base;
    }

    return temp;
}

arma::mat generate_op_matrix(uint index, operators_data *ops, xxz_real *ham, double epsilon)
{
    /*
        This function generates matrix of operator using operators_info(index,...)
        and eigenvectors of the hamiltonian to represent them in basis of
        hamiltonian eigenstates.     
    */
    uint64_t states = ham->get_n_states();
    arma::mat op(states, states); // initialize matrix of the operator
    op.zeros();

    if (ops->trans_sym)
    {
        double mult = 1.0;

        for (uint shift = 0; shift < ops->sites; shift++)
        {
            for (uint i = 0; i < states; i++)
            {
                binary_rep final_state = ham->get_state_binary(i);
                uint first_site_index = ops->operators_info(index, 0) + shift;
                double mat_el = 1.0; // matrix element

                for (uint ii = 0; ii < ops->max_supp; ii++)
                {
                    uint site_index = first_site_index + ii;
                    if (site_index >= ops->sites)
                        site_index -= ops->sites;

                    switch (ops->operators_info(index, ii + 1))
                    {
                    case 0: // Identity operator
                    {
                        // Do nothing
                        break;
                    }
                    case 1: // S^+_{site_index}
                    {
                        double ap;
                        if (final_state.state[site_index])
                        {
                            ap = 0.0;
                            //do not change final state
                        }
                        else
                        {
                            ap = 1;
                            final_state.state[site_index] = true; // Modify final state
                        }
                        mat_el *= ap;
                        break;
                    }
                    case 2: // S^z_{site_index}
                    {
                        double sz = 0;
                        if (final_state.state[site_index])
                            sz = 1;
                        mat_el *= (sz - 0.5) * sqrt(2.0);
                        // Do not change final state
                        break;
                    }
                    case 3: // S^-_{site_index}
                    {
                        double am;
                        if (final_state.state[site_index])
                        {
                            am = 1;
                            final_state.state[site_index] = false; // Modify final state
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
                        std::cout << "#There is no such operator.\n Someting went wrong.\n";
                        break;
                    }
                } // end of support loop

                if (std::abs(mat_el) < 1e-6)
                    continue; // check if mat_el is zero

                uint64_t num;
                ham->get_basis()->fast_b2d(&num, final_state.state);
                op(i, ham->reverse_index(num)) += mult * mat_el;

            } // end of basis states loop
            if (ops->flip)
                mult *= -1.0;
        } // end of shifts loop
    }
    else
    {
        for (uint i = 0; i < states; i++)
        {
            binary_rep final_state = ham->get_state_binary(i);
            uint first_site_index = ops->operators_info(index, 0);
            double mat_el = 1.0; // matrix element

            for (uint ii = 0; ii < ops->max_supp; ii++)
            {
                uint site_index = first_site_index + ii;
                if (site_index >= ops->sites)
                    site_index -= ops->sites;

                switch (ops->operators_info(index, ii + 1))
                {
                case 0: // Identity operator
                {
                    // Do nothing
                    break;
                }
                case 1: // S^+_{site_index}
                {
                    double ap;
                    if (final_state.state[site_index])
                    {
                        ap = 0.0;
                        //do not change final state
                    }
                    else
                    {
                        ap = 1;
                        final_state.state[site_index] = true; // Modify final state
                    }
                    mat_el *= ap;
                    break;
                }
                case 2: // S^z_{site_index}
                {
                    double sz = 0;
                    if (final_state.state[site_index])
                        sz = 1;
                    mat_el *= (sz - 0.5) * sqrt(2.0);
                    // Do not change final state
                    break;
                }
                case 3: // S^-_{site_index}
                {
                    double am;
                    if (final_state.state[site_index])
                    {
                        am = 1;
                        final_state.state[site_index] = false; // Modify final state
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
                    std::cout << "#There is no such operator.\n Someting went wrong.\n";
                    break;
                }
            } // end of support loop

            if (std::abs(mat_el) < 1e-6)
                continue; // check if mat_el is zero

            uint64_t num;
            ham->get_basis()->fast_b2d(&num, final_state.state);
            op(i, ham->reverse_index(num)) += mat_el;

        } // end of basis states loop
    }

        // add/substract Hermitian conjugate - actually just transpose because this matrix is real
        if (ops->flag)
            op += arma::trans(op); // O+O^dag
        else
            op -= arma::trans(op); // i(O-O^dag), i is irrelevant for H-S product

        // transform op matrix to basis of Hamiltonian eigenstates op --> eigvec^T.op.eigvec
        op = arma::trans(*ham->get_eigenvectors()) * op * (*ham->get_eigenvectors());

        // Traceless
        double trace = arma::trace(op) / states;
        op.diag() -= trace;

        // Normed
        double norm = sqrt(arma::trace(op.t() * op) / states);
        op /= norm;

        // Time averaging

        if (epsilon < 1e5)
        {
            #pragma omp parallel for
            for (uint j = 0; j < states; j++)
            {
                for (uint jj = 0; jj < states; jj++)
                {
                    if (std::abs(ham->get_eigenvalue(j) - ham->get_eigenvalue(jj)) > epsilon)
                    {
                        op(j, jj) = 0.0;
                    }
                }
            }
        }

    return op;
}

void usage(bool flag, char *progname)
{
    if (!flag)
    {
        printf("%s: [-h] [-option]\n", progname);
    }
    else
    {
        printf("%s: [-h] [-option] \n", progname);
        printf("Available options:\n");
        printf("-t -> hopping integral\n");
        printf("-d -> delta, interaction strength\n");
        printf("-a -> alpha, integrability breaking\n");
        printf("-s -> number of sites\n");
        printf("-m -> operators support\n");
        printf("-T -> 1 to turn on translational symmetry, 0 to turn off\n");
        printf("-F -> 1 to turn on sign flip in translational symmetry, 0 to turn off\n");
        printf("-C -> 1 to generate commuting operators, 0 to generate noncommuting\n");
        printf("-G -> 1 to generate O+O^+, 0 to generate i(O-O^+)\n");
        printf("-S -> 0 - does not save matrix elements, i!=0 - saves matrix elements of eigen operator with i-th largers stiffness\n");
        printf("Default: -t -0.5 -d 1.0 -a 0.0 -s 8 -m 3 -G 1 -T 0 -F 0 -C 0 -S 0\n");
    }
    exit(1);
}

arma::mat generate_op_matrix_from_vector(arma::vec coefficients, operators_data *ops, xxz_real *ham, double epsilon)
{

    arma::mat op(ham->get_n_states(), ham->get_n_states());
    op.zeros();
    for (uint i = 0; i < ops->no_op; i++)
    {
        if (std::abs(coefficients[i]) < 1e-15)
            continue;
        op += coefficients[i] * generate_op_matrix(i, ops, ham, epsilon);
    }

    return op;
}

void save_op_for_ham(uint save_op, operators_data *ops, xxz_real *ham, arma::vec *eigenvalues, arma::mat *eigenvectors)
{
    if (!ham->is_diag())
        ham->diagonalize();
    std::cout << "# Saving operator with lambda = " << (*eigenvalues)(ops->no_op - save_op) << " for alpha " << ham->get_alpha() << std::endl;
    arma::mat opmat = generate_op_matrix_from_vector(eigenvectors->col(ops->no_op - save_op), ops, ham, 1e+8);
    std::cout << "# lambda test = " << arma::trace(opmat.t() * opmat) / ham->get_n_states() << std::endl;
    arma::mat out(2, ham->get_n_states() * ham->get_n_states());
    int index = 0;
    for (uint n = 0; n < ham->get_n_states(); n++)
    {
        for (uint m = 0; m < ham->get_n_states(); m++)
        {
            out(0, index) = ham->get_eigenvalue(n) - ham->get_eigenvalue(m);
            out(1, index) = opmat(n, m);
            index++;
        }
    }
    out = arma::trans(out);
    std::string alpha_str = std::to_string(ham->get_alpha()).substr(0, std::to_string(ham->get_alpha()).find(".") + 3);
    std::string t_str = std::to_string(ham->get_t()).substr(0, std::to_string(ham->get_t()).find(".") + 2);
    std::string d_str = std::to_string(ham->get_delta()).substr(0, std::to_string(ham->get_delta()).find(".") + 2);
    std::string filename = "spin_t_" + t_str + "_d_" + d_str + "_L_" + std::to_string(ops->sites) + "_alpha_" + alpha_str;
    out.save(filename + ".csv", arma::csv_ascii);
}