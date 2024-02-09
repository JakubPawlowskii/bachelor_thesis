#include "combinadics.hpp"  
	
combinadics::combinadics(uint sites, uint particles):
n_sites(sites), n_particles(particles)
{
   	n_states = binomial_coeff(n_sites, n_particles);
	max = max_state(n_sites, n_particles);
	min = min_state(n_particles);
	cnt = 0;
	states = new uint64_t[n_states];

    do
	{
		states[cnt] = min;
		cnt++;
	}
	while (next_combination(min) && min <= max);
	

}
	
combinadics::combinadics()
{
    combinadics(2,1);
}
combinadics::combinadics(const combinadics& obj)
{
    n_particles = obj.n_particles;
    n_sites = obj.n_sites;
    n_states = obj.n_states;
    min = obj.min;
    max = obj.max;
    cnt = obj.cnt;
    states = new uint64_t[n_states];
    *states = *obj.states;

}


combinadics::~combinadics()
{
	delete[] states;
}

bool combinadics::next_combination(uint64_t& x) // assume x has form x'01^a10^b in binary
{
	uint64_t u = x & -x; // extract rightmost bit 1; u =  0'00^a10^b
	uint64_t v = u + x; // set last non-trailing bit 0, and clear to the right; v=x'10^a00^b
	if (v == 0) // then overflow in v, or x==0
		return false; // signal that next k-combination cannot be represented
	x = v + (((v ^ x) / u) >> 2); // v^x = 0'11^a10^b, (v^x)/u = 0'0^b1^{a+2}, and x ← x'100^b1^a
	return true; // successful completion
}

uint64_t combinadics::fast_power(uint64_t base, uint64_t power)
{
	uint64_t result = 1;
	while (power > 0)
	{
		//if(power % 2 == 1) { 
		if (power & 1)
		{
			result = (result * base); //% MOD;
		}
		base = (base * base); //% MOD;
		//power = power / 2;
		power >>= 1;
	}
	return result;
}


uint64_t combinadics::binomial_coeff(int n, int k)
{
	uint64_t res = 1;

	// Since C(n, k) = C(n, n-k) 
	if (k > n - k)
		k = n - k;

	// Calculate value of 
	// [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1] 
	for (int i = 0; i < k; ++i)
	{
		res *= (n - i);
		res /= (i + 1);
	}

	return res;
}


uint64_t combinadics::max_state(int n_sites, int n_particles)
{
	uint64_t max = 0;
	int cnt = 0;

	while (cnt < n_particles)
	{
		max += fast_power(2, n_sites - 1);
		n_sites--;
		cnt++;;
	}
	return max;
}

uint64_t combinadics::min_state(int n_particles)
{
	uint64_t min = 0;
	int cnt = 0;

	while (cnt < n_particles)
	{
		min += fast_power(2, cnt);
		cnt++;
	}
	return min;
}

bool combinadics::check_bit(uint64_t state_idx, uint64_t pos)
{
    try
    {
        if(state_idx > n_states) throw state_idx;
    }
    catch(uint64_t idx)
    {
        std::cerr << "Index " << idx << " in check_bit(uint64_t state_idx, uint64_t pos) is larger than number of states (" << n_states << ")\n";
        std::cerr << "Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    return static_cast<bool>((get_state(state_idx)) & (1<<(pos)));
}

bool check_bit_from_num(uint64_t num, uint64_t pos)
{
    return static_cast<bool>((num) & (1<<(pos)));

}


uint64_t* combinadics::get_all_states() const
{
    return states;
}
uint combinadics::get_n_sites() const
{
    return n_sites;
}
uint combinadics::get_n_particles() const
{
    return n_particles;
}

uint combinadics::get_n_states() const
{
    return n_states;
}


uint64_t combinadics::get_state(uint64_t idx) const
{
    try
    {
        if(idx > n_states) throw idx;
    }
    catch(uint64_t idx)
    {
        std::cerr << "Index " << idx << " in get_state(uint64_t idx) is larger than number of states (" << n_states << ")\n";
        std::cerr << "Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    return states[idx];
}

binary_rep combinadics::get_state_binary(uint64_t idx)
{
    try
    {
        if(idx > n_states) throw idx;
    }
    catch(uint64_t idx)
    {
        std::cerr << "Index " << idx << " in get_state_binary(uint64_t idx) is larger than number of states (" << n_states << ")\n";
        std::cerr << "Aborting..." << std::endl;
        exit(EXIT_FAILURE);
    }

    uint64_t state = states[idx];
    bool* res = new bool[n_sites];
    fast_d2b(state, res);
    binary_rep bin_state = {res, n_sites};

    return bin_state;
}

// /* decimal to binary */
void combinadics::fast_d2b(uint64_t x, bool* c)
{
	for (uint i = 0; i < n_sites; i++) c[i] = static_cast<bool>((x >> i) & 0x1);
		// c->set(i, (x >> i) & 0x1);
}

/*
 Binary to decimal
*/
void combinadics::fast_b2d(uint64_t* n, bool* c)
{
	int i = n_sites;
	*n = 0;
	while (i--)
	{
		*n <<= 1;
		*n += *(c + i);
	}
}

void combinadics::print_states()
{

    std::cout<<"Printing all states for " << n_sites << " sites and " << n_particles << " particles:\n";
    for(uint i = 0; i < n_states; i++)
    {
        std::cout<< get_state_binary(i) << " ---- " << get_state(i) << "\n";
    }
	if (n_states == 1) std::cout << "There is " << n_states << " state." << std::endl;
	else std::cout << "There are " << n_states << " states." << std::endl;

}

uint64_t combinadics::reverse_index(uint64_t state)
{
	for(uint i = 0; i < n_states; i++)
	{
		if(states[i] == state) return i;
	}
	std::cerr << "This state is not in the basis (reverse_index).\n";
	exit(EXIT_FAILURE);
}


std::ostream& operator<<(std::ostream& os, const binary_rep obj)
{
    for(uint i = 0; i < obj.sites; i++)
	{
		os << obj.state[i];
	}
    
    return os;
}

