#include "xxz_real.hpp"

// xxz_real::xxz_real(uint64_t sites, double t, double delta):
// t(t), delta(delta)
// {
//     type = full;
//     name = "integrable XXZ hamiltonian with full Hilbert space";
//     n_sites = sites;
//     combinadics bit_functions(n_sites,1);
//     n_states = bit_functions.fast_power(2,sites);
//     // std::cout << n_states << std::endl;

//     ham.zeros(n_states,n_states);

//     for (uint64_t i = 0; i < n_states; i++)
//     {
//         bool* state = new bool[n_sites];
//         bit_functions.fast_d2b(i, state);
//         // binary_rep ss = {state, n_sites};
//         // std::cout << ss << std::endl;
//         for (uint ii = 0; ii < n_sites; ii++)
//         {
//             // double sign = 1.0;
//             uint first = ii;
//             uint second = ii + 1;
//             if (second == n_sites)
//                 second = 0;
//             bool s1 = state[first];
//             bool s2 = state[second];

//             ham(i, i) += delta * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s2) - 0.5);
//          if (s1 == false && s2 == true)
//             {
//                 state[first] = true;
//                 state[second] = false;
//                 uint64_t num;
//                 bit_functions.fast_b2d(&num, state);
//                 ham(i, num) += -t;
//                 state[first] = false;
//                 state[second] = true;
//             }
//             if (s1 == true && s2 == false)
//             {
//                 state[first] = false;
//                 state[second] = true;
//                 uint64_t num;
//                 bit_functions.fast_b2d(&num, state);
//                 ham(i, num) += -t;
//                 state[first] = true;
//                 state[second] = false;
//             }
//         }
//     }
//     if(!ham.is_symmetric())
//     {
//         std::cerr << "Obtained hamiltonian is not symmetric. Aborting..." << std::endl;
//         exit(EXIT_FAILURE);
//     }
// }

xxz_real::xxz_real(uint64_t sites, double t, double delta, double alpha) : t(t), delta(delta), alpha(alpha)
{

    type = full;
    name = "XXZ hamiltonian with full Hilbert space";
    n_sites = sites;
    // combinadics bit_functions(n_sites,0);
    basis = new combinadics(n_sites, 0);
    n_states = basis->fast_power(2, sites);
    // std::cout << n_states << std::endl;

    ham.zeros(n_states, n_states);

    for (uint64_t i = 0; i < n_states; i++)
    {
        bool *state = new bool[n_sites];
        basis->fast_d2b(i, state);
        // binary_rep ss = {state, n_sites};
        // std::cout << ss << std::endl;
        for (uint ii = 0; ii < n_sites; ii++)
        {
            // double sign = 1.0;
            uint first = ii;
            uint second = ii + 1;
            uint third = ii + 2;
            if (second == n_sites)
                second = 0;
            if (third >= n_sites)
                third = third - n_sites;

            bool s1 = state[first];
            bool s2 = state[second];
            bool s3 = state[third];

            ham(i, i) += delta * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s2) - 0.5);
            ham(i, i) += alpha * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s3) - 0.5);

            if (s1 == false && s2 == true)
            {

                state[first] = true;
                state[second] = false;
                uint64_t num;
                basis->fast_b2d(&num, state);
                ham(i, num) += -t;
                state[first] = false;
                state[second] = true;
            }
            if (s1 == true && s2 == false)
            {
                state[first] = false;
                state[second] = true;
                uint64_t num;
                basis->fast_b2d(&num, state);
                ham(i, num) += -t;
                state[first] = true;
                state[second] = false;
            }
        }
        delete[] state;
    }
    if (!ham.is_symmetric())
    {
        std::cerr << "Obtained hamiltonian is not symmetric. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    constructed = true;
}

xxz_real::xxz_real(uint64_t sites, uint64_t particles, double t, double delta, double alpha) : t(t), delta(delta), alpha(alpha)
{
    type = const_particles;
    name = "XXZ hamiltonian with fixed particle number subspace";
    n_sites = sites;
    n_particles = particles;
    basis = new combinadics(n_sites, particles);
    n_states = basis->get_n_states();

    // std::cout << n_states << std::endl;

    ham.zeros(n_states, n_states);

    for (uint64_t i = 0; i < n_states; i++)
    {
        binary_rep state = basis->get_state_binary(i);
        // binary_rep ss = {state, n_sites};
        // std::cout << ss << std::endl;
        for (uint ii = 0; ii < n_sites; ii++)
        {
            // double sign = 1.0;
            uint first = ii;
            uint second = ii + 1;
            uint third = ii + 2;

            if (second == n_sites)
                second = 0;
            if (third >= n_sites)
                third = third - n_sites;

            bool s1 = state.state[first];
            bool s2 = state.state[second];
            bool s3 = state.state[third];

            ham(i, i) += delta * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s2) - 0.5);
            ham(i, i) += alpha * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s3) - 0.5);

            if (s1 == false && s2 == true)
            {

                state.state[first] = true;
                state.state[second] = false;
                uint64_t num;
                basis->fast_b2d(&num, state.state);
                ham(i, basis->reverse_index(num)) += -t;
                state.state[first] = false;
                state.state[second] = true;
            }
            if (s1 == true && s2 == false)
            {
                state.state[first] = false;
                state.state[second] = true;
                uint64_t num;
                basis->fast_b2d(&num, state.state);
                ham(i, basis->reverse_index(num)) += -t;
                state.state[first] = true;
                state.state[second] = false;
            }
        }
    }
    if (!ham.is_symmetric())
    {
        std::cerr << "Obtained hamiltonian is not symmetric. Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    constructed = true;
}

void xxz_real::regenerate()
{

    if (type == full)
    {
    
        ham.zeros(n_states, n_states);

        for (uint64_t i = 0; i < n_states; i++)
        {
            bool *state = new bool[n_sites];
            basis->fast_d2b(i, state);
            // binary_rep ss = {state, n_sites};
            // std::cout << ss << std::endl;
            for (uint ii = 0; ii < n_sites; ii++)
            {
                // double sign = 1.0;
                uint first = ii;
                uint second = ii + 1;
                uint third = ii + 2;
                if (second == n_sites)
                    second = 0;
                if (third >= n_sites)
                    third = third - n_sites;

                bool s1 = state[first];
                bool s2 = state[second];
                bool s3 = state[third];

                ham(i, i) += delta * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s2) - 0.5);
                ham(i, i) += alpha * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s3) - 0.5);

                if (s1 == false && s2 == true)
                {

                    state[first] = true;
                    state[second] = false;
                    uint64_t num;
                    basis->fast_b2d(&num, state);
                    ham(i, num) += -t;
                    state[first] = false;
                    state[second] = true;
                }
                if (s1 == true && s2 == false)
                {
                    state[first] = false;
                    state[second] = true;
                    uint64_t num;
                    basis->fast_b2d(&num, state);
                    ham(i, num) += -t;
                    state[first] = true;
                    state[second] = false;
                }
            }
            delete[] state;
        }
    }
    else if (type == const_particles)
    {
        ham.zeros(n_states, n_states);

        for (uint64_t i = 0; i < n_states; i++)
        {
            binary_rep state = basis->get_state_binary(i);
            // binary_rep ss = {state, n_sites};
            // std::cout << ss << std::endl;
            for (uint ii = 0; ii < n_sites; ii++)
            {
                // double sign = 1.0;
                uint first = ii;
                uint second = ii + 1;
                uint third = ii + 2;

                if (second == n_sites)
                    second = 0;
                if (third >= n_sites)
                    third = third - n_sites;

                bool s1 = state.state[first];
                bool s2 = state.state[second];
                bool s3 = state.state[third];

                ham(i, i) += delta * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s2) - 0.5);
                ham(i, i) += alpha * (static_cast<double>(s1) - 0.5) * (static_cast<double>(s3) - 0.5);

                if (s1 == false && s2 == true)
                {

                    state.state[first] = true;
                    state.state[second] = false;
                    uint64_t num;
                    basis->fast_b2d(&num, state.state);
                    ham(i, basis->reverse_index(num)) += -t;
                    state.state[first] = false;
                    state.state[second] = true;
                }
                if (s1 == true && s2 == false)
                {
                    state.state[first] = false;
                    state.state[second] = true;
                    uint64_t num;
                    basis->fast_b2d(&num, state.state);
                    ham(i, basis->reverse_index(num)) += -t;
                    state.state[first] = true;
                    state.state[second] = false;
                }
            }
        }
    }
    constructed = true;
}

double xxz_real::get_alpha() const{
    return alpha;
}
double xxz_real::get_t() const{
    return t;
}
double xxz_real::get_delta() const{
    return delta;
}