/*
*	Small library for combinatorial number system based on Gosper's hack
*
*/


#ifndef COMBINADICS_H
#define COMBINADICS_H
#pragma once

#include <iostream>

struct binary_rep
{
	bool* state;
	uint64_t sites;
};


class combinadics  
{
	private:
		unsigned int n_sites;
		unsigned int n_particles;
		unsigned int n_states;
		uint64_t *states;
		uint64_t min;
		uint64_t max;
		uint64_t cnt;
		
		bool next_combination(uint64_t& x);
		uint64_t binomial_coeff(int n, int k);
		uint64_t max_state(int n_sites, int n_particles);
		uint64_t min_state(int n_particles);
		


	public:
		combinadics();
		combinadics(const combinadics& obj);
		combinadics(uint sites, uint particles);
		~combinadics();
		bool check_bit(uint64_t state_idx, uint64_t pos);
		bool check_bit_from_num(uint64_t num, uint64_t pos);
		uint64_t* get_all_states() const;
		uint get_n_sites() const;
		uint get_n_particles() const;
		uint get_n_states() const;
		uint64_t get_state(uint64_t idx) const;
		binary_rep get_state_binary(uint64_t idx);
		void print_states();
		void fast_d2b(uint64_t x, bool* c);
		void fast_b2d(uint64_t* n, bool* c);
 		uint64_t fast_power(uint64_t base, uint64_t power);
		uint64_t reverse_index(uint64_t state);


};

std::ostream& operator<<(std::ostream& os, const binary_rep obj);

#endif


