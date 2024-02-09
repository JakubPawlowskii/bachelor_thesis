#include <iostream>
#include <armadillo>
#include <bitset>
#include <cmath>
#include <iomanip>
using std::cout;
using std::endl;

namespace parameters
{
    constexpr uint sites = 10;       //var1     // no. of sites
    uint states = pow(2, sites);     // no. of states, 2^sites
    constexpr double delta = -0.5;   //var2 // delta in Heisenberg model
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
} // namespace operators

typedef std::bitset<parameters::sites> bits;
arma::wall_clock timer;

arma::uvec append_ops(int begin, int end, arma::uvec ops);
int generate_hamiltonian_dense();
int generate_operators();
arma::uvec dec_to_base(uint); // converts decimal number to base specified by number of "basic" operators
arma::mat generate_op_matrix(int);

int main()
{

    using namespace parameters;
    using namespace operators;

    timer.tic();
    cout << "#Fermion code\n";
    generate_hamiltonian_dense();
    generate_operators();

    for (uint i = 0; i < no_op; i++)
    {
        cout << i + 1 << "   " << operators_info(i, 0) + 1 << " "; // << operators_info(i, 1) << operators_info(i, 2) << operators_info(i, 3) << endl;
        for (uint j = 1; j <= max_supp; j++)
            cout << operators_info(i, j);
        cout << endl;
        if ((i + 1) / sites * sites == i + 1)
            cout << endl;
    }
    cout << "#" << no_op << " operators generated." << endl;

    /*
        Instead of full correlation matrix, generate matrix of one desired operator
        and compute its norm.
    */
    arma::mat op(states, states);
    op.zeros();

    arma::uvec op_def = {};
    arma::vec coeffs;
    arma::vec coeffs_vals;
    float norm_correction = 1.0;
    if (max_supp == 3)
    {

        if ((sites == 8) && delta > 0.0)
        {
            op_def = {8, 9, 10, 11, 12, 13, 14, 15, 72, 73, 74, 75, 76, 77, 78, 79}; // L = 8
            double coeff1 = -0.243024;
            double coeff2 = 0.256786;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
        if ((sites == 10) && delta > 0.0)
        {
            op_def = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99}; // L = 10
            double coeff1 = 0.225545;
            double coeff2 = -0.221652;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
        if ((sites == 12) && delta > 0.0)
        {
            op_def = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119}; // L = 12
            double coeff1 = -0.208924;
            double coeff2 = 0.199209;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
        if ((sites == 14) && delta > 0.0)
        {
            norm_correction = 12.0 / 14.0;
            op_def = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 126, 127,
                      128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139}; // L = 14
            double coeff1 = -0.208924;
            double coeff2 = 0.199209;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
        if ((sites == 16) && delta > 0.0)
        {
            norm_correction = 12.0 / 16.0;
            op_def = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 144, 145, 146,
                      147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159}; // L = 16
            double coeff1 = -0.208924;
            double coeff2 = 0.199209;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
        //================================================================================================

        if ((sites == 8) && delta < 0.0)
        {
            op_def = {8, 9, 10, 11, 12, 13, 14, 15, 72, 73, 74, 75, 76, 77, 78, 79}; // L = 8
            double coeff1 = 0.283907;
            double coeff2 = 0.210705;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
        if ((sites == 10) && delta < 0.0)
        {
            op_def = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}; // L = 10
            double coeff1 = 0.316228;

            coeffs.resize(sites);
            float mult = 1.0;
            for (uint i = 0; i < sites; i++)
            {
                coeffs[i] = mult * coeff1;
                mult *= -1.0;
            }
        }
        if ((sites == 12) && delta < 0.0)
        {
            op_def = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119}; // L = 12
            double coeff1 = -0.248631;
            double coeff2 = -0.146683;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
        if ((sites == 14) && delta < 0.0)
        {
            norm_correction = 12.0 / 14.0;
            op_def = {
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
            }; // L = 14
            double coeff1 = 0.316228;

            coeffs.resize(sites);
            float mult = 1.0;
            for (uint i = 0; i < sites; i++)
            {
                coeffs[i] = mult * coeff1;
                mult *= -1.0;
            }
        }
        if ((sites == 16) && delta < 0.0)
        {
            norm_correction = 12.0 / 16.0;
            op_def = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 144, 145, 146,
                      147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159}; // L = 16
            double coeff1 = -0.248631;
            double coeff2 = -0.146683;
            coeffs.resize(2 * sites);
            for (uint i = 0; i < sites; i++)
                coeffs[i] = coeff1;
            for (uint i = sites; i < 2 * sites; i++)
                coeffs[i] = coeff2;
        }
    }
    if (max_supp == 4)
    {
        if ((sites == 8) && delta > 0.0)
        {
            op_def = append_ops(8, 15, op_def); // L = 8
            op_def = append_ops(72, 79, op_def);
            op_def = append_ops(136, 143, op_def);
            op_def = append_ops(160, 167, op_def);
            op_def = append_ops(248, 255, op_def);
            op_def = append_ops(264, 271, op_def);
            op_def = append_ops(280, 287, op_def);
            op_def = append_ops(336, 343, op_def);
            op_def = append_ops(384, 391, op_def);
            op_def = append_ops(424, 431, op_def);
            op_def = append_ops(464, 471, op_def);
            op_def = append_ops(488, 495, op_def);
            op_def = append_ops(544, 551, op_def);

            coeffs_vals = {0.201409, -0.205771, 0.000811041, 0.00355081, 0.00132639, 0.00564472,
                           -0.00176694, 0.204923, 0.00176694, -0.00132639, 0.00355081, 0.000370495, 0.00564472};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
        if ((sites == 10) && delta > 0.0)
        {
            op_def = append_ops(10, 19, op_def); // L = 8
            op_def = append_ops(90, 99, op_def);
            op_def = append_ops(170, 179, op_def);
            op_def = append_ops(200, 209, op_def);
            op_def = append_ops(310, 319, op_def);
            op_def = append_ops(330, 339, op_def);
            op_def = append_ops(350, 359, op_def);
            op_def = append_ops(420, 429, op_def);
            op_def = append_ops(480, 489, op_def);
            op_def = append_ops(530, 539, op_def);
            op_def = append_ops(580, 589, op_def);
            op_def = append_ops(610, 619, op_def);
            op_def = append_ops(680, 689, op_def);

            coeffs_vals = {0.186111, -0.183443, -0.00150131, -0.00116627, 0.00527862, 0.00223174, 0.000508927,
                           0.177826, -0.000508927, -0.00527862, -0.00116627, 0.00428623, 0.00223174};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
        if ((sites == 12) && delta > 0.0)
        {
            norm_correction = 10.0 / 12.0;
            op_def = append_ops(12, 23, op_def); // L = 8
            op_def = append_ops(108, 119, op_def);
            op_def = append_ops(204, 215, op_def);
            op_def = append_ops(240, 251, op_def);
            op_def = append_ops(372, 383, op_def);
            op_def = append_ops(396, 407, op_def);
            op_def = append_ops(420, 431, op_def);
            op_def = append_ops(504, 515, op_def);
            op_def = append_ops(576, 587, op_def);
            op_def = append_ops(636, 647, op_def);
            op_def = append_ops(696, 707, op_def);
            op_def = append_ops(732, 743, op_def);
            op_def = append_ops(816, 827, op_def);

            coeffs_vals = {0.186111, -0.183443, -0.00150131, -0.00116627, 0.00527862, 0.00223174, 0.000508927,
                           0.177826, -0.000508927, -0.00527862, -0.00116627, 0.00428623, 0.00223174};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
        if ((sites == 14) && delta > 0.0)
        {
            norm_correction = 10.0 / 14.0;
            op_def = append_ops(14, 27, op_def); // L = 8
            op_def = append_ops(126, 139, op_def);
            op_def = append_ops(238, 251, op_def);
            op_def = append_ops(280, 293, op_def);
            op_def = append_ops(434, 447, op_def);
            op_def = append_ops(462, 475, op_def);
            op_def = append_ops(490, 503, op_def);
            op_def = append_ops(588, 601, op_def);
            op_def = append_ops(672, 685, op_def);
            op_def = append_ops(742, 755, op_def);
            op_def = append_ops(812, 825, op_def);
            op_def = append_ops(854, 867, op_def);
            op_def = append_ops(952, 965, op_def);

            coeffs_vals = {0.186111, -0.183443, -0.00150131, -0.00116627, 0.00527862, 0.00223174, 0.000508927,
                           0.177826, -0.000508927, -0.00527862, -0.00116627, 0.00428623, 0.00223174};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
        if ((sites == 16) && delta > 0.0)
        {
            norm_correction = 10.0 / 16.0;
            op_def = append_ops(16, 31, op_def);
            op_def = append_ops(144, 159, op_def);
            op_def = append_ops(272, 287, op_def);
            op_def = append_ops(320, 335, op_def);
            op_def = append_ops(496, 511, op_def);
            op_def = append_ops(528, 543, op_def);
            op_def = append_ops(560, 575, op_def);
            op_def = append_ops(672, 687, op_def);
            op_def = append_ops(768, 783, op_def);
            op_def = append_ops(848, 863, op_def);
            op_def = append_ops(928, 943, op_def);
            op_def = append_ops(976, 991, op_def);
            op_def = append_ops(1088, 1103, op_def);

            coeffs_vals = {0.186111, -0.183443, -0.00150131, -0.00116627, 0.00527862, 0.00223174, 0.000508927,
                           0.177826, -0.000508927, -0.00527862, -0.00116627, 0.00428623, 0.00223174};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
        //================================================================================================

        if ((sites == 8) && delta < 0.0)
        {
            op_def = append_ops(8, 15, op_def); // L = 8
            op_def = append_ops(72, 79, op_def);
            op_def = append_ops(136, 143, op_def);
            op_def = append_ops(160, 167, op_def);
            op_def = append_ops(248, 255, op_def);
            op_def = append_ops(264, 271, op_def);
            op_def = append_ops(280, 287, op_def);
            op_def = append_ops(336, 343, op_def);
            op_def = append_ops(384, 391, op_def);
            op_def = append_ops(424, 431, op_def);
            op_def = append_ops(464, 471, op_def);
            op_def = append_ops(488, 495, op_def);
            op_def = append_ops(544, 551, op_def);

            coeffs_vals = {-0.229406, -0.0825276, 0.198886, 0.0214865, 0.0123395, 0.0711049,
                           0.0192924, -0.109458, -0.0192924, -0.0123395, 0.0214865, 0.0440604, 0.0711049};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
        if ((sites == 10) && delta < 0.0)
        {
            op_def = append_ops(10, 19, op_def); // L = 8
            op_def = append_ops(170, 179, op_def);
            op_def = append_ops(200, 209, op_def);
            op_def = append_ops(310, 319, op_def);
            op_def = append_ops(330, 339, op_def);
            op_def = append_ops(350, 359, op_def);
            op_def = append_ops(420, 429, op_def);
            op_def = append_ops(480, 489, op_def);
            op_def = append_ops(530, 539, op_def);
            op_def = append_ops(580, 589, op_def);
            op_def = append_ops(610, 619, op_def);
            op_def = append_ops(680, 689, op_def);

            coeffs_vals = {0.270452, -0.0930616, -0.0152635, -0.0420037, -0.0489893, -0.0496447,
                           0.0355607, 0.0496447, 0.0420037, -0.0152635, 0.0566287, -0.0489893};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
            {
                float mult = 1.0;

                for (uint i = j * sites; i < (j + 1) * sites; i++)
                {
                    coeffs[i] = mult * coeffs_vals[j];
                    mult *= -1.0;
                }
            }
        }

        if ((sites == 12) && delta < 0.0)
        {
            norm_correction = 10.0 / 12.0;
            op_def = append_ops(12, 23, op_def); // L = 8
            op_def = append_ops(108, 119, op_def);
            op_def = append_ops(204, 215, op_def);
            op_def = append_ops(240, 252, op_def);
            op_def = append_ops(372, 383, op_def);
            op_def = append_ops(396, 407, op_def);
            op_def = append_ops(420, 431, op_def);
            op_def = append_ops(504, 515, op_def);
            op_def = append_ops(576, 587, op_def);
            op_def = append_ops(636, 647, op_def);
            op_def = append_ops(696, 707, op_def);
            op_def = append_ops(732, 743, op_def);
            op_def = append_ops(816, 827, op_def);

            coeffs_vals = {-0.229406, -0.0825276, 0.198886, 0.0214865, 0.0123395, 0.0711049,
                           0.0192924, -0.109458, -0.0192924, -0.0123395, 0.0214865, 0.0440604, 0.0711049};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
        if ((sites == 14) && delta < 0.0)
        {
            norm_correction = 10.0 / 14.0;
            op_def = append_ops(14, 27, op_def); // L = 8
            op_def = append_ops(238, 251, op_def);
            op_def = append_ops(280, 293, op_def);
            op_def = append_ops(434, 447, op_def);
            op_def = append_ops(462, 475, op_def);
            op_def = append_ops(490, 503, op_def);
            op_def = append_ops(588, 601, op_def);
            op_def = append_ops(672, 685, op_def);
            op_def = append_ops(742, 755, op_def);
            op_def = append_ops(812, 825, op_def);
            op_def = append_ops(854, 867, op_def);
            op_def = append_ops(952, 965, op_def);

            coeffs_vals = {0.270452, -0.0930616, -0.0152635, -0.0420037, -0.0489893, -0.0496447,
                           0.0355607, 0.0496447, 0.0420037, -0.0152635, 0.0566287, -0.0489893};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
            {
                float mult = 1.0;

                for (uint i = j * sites; i < (j + 1) * sites; i++)
                {
                    coeffs[i] = mult * coeffs_vals[j];
                    mult *= -1.0;
                }
            }
        }
        if ((sites == 16) && delta < 0.0)
        {
            norm_correction = 10.0 / 16.0;
            op_def = append_ops(16, 31, op_def); // L = 8
            op_def = append_ops(144, 159, op_def);
            op_def = append_ops(272, 287, op_def);
            op_def = append_ops(320, 335, op_def);
            op_def = append_ops(496, 511, op_def);
            op_def = append_ops(528, 543, op_def);
            op_def = append_ops(560, 575, op_def);
            op_def = append_ops(672, 687, op_def);
            op_def = append_ops(768, 783, op_def);
            op_def = append_ops(848, 863, op_def);
            op_def = append_ops(928, 943, op_def);
            op_def = append_ops(976, 991, op_def);
            op_def = append_ops(1088, 1103, op_def);

            coeffs_vals = {-0.229406, -0.0825276, 0.198886, 0.0214865, 0.0123395, 0.0711049,
                           0.0192924, -0.109458, -0.0192924, -0.0123395, 0.0214865, 0.0440604, 0.0711049};
            coeffs.resize(coeffs_vals.size() * sites);

            for (uint j = 0; j < coeffs_vals.size(); j++)
                for (uint i = j * sites; i < (j + 1) * sites; i++)
                    coeffs[i] = coeffs_vals[j];
        }
    }
    int ind = 0;
    for (auto num : op_def)
    {
        op += coeffs.at(ind) * generate_op_matrix(num);
        ind++;
        cout << num << "\n";
    }
    auto eigval = arma::trace(op.t() * op) / states;
    cout << "\n"
         << std::setprecision(17) << eigval << "\n";
    cout << norm_correction * eigval << "\n";

    cout << std::setprecision(5) << "#Execution time: " << timer.toc() << " s" << endl;
    return 0;
}

arma::uvec append_ops(int begin, int end, arma::uvec ops)
{
    int add_len = end - begin + 1;

    arma::uvec res(ops.size() + add_len);
    for (uint i = 0; i < ops.size(); i++)
    {
        res[i] = ops[i];
    }
    int j = 0;
    for (uint i = ops.size(); i < res.size(); i++)
    {
        res[i] = begin + j;
        j++;
    }
    return res;
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
arma::mat generate_op_matrix(int index)
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
