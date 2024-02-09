#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H
#pragma once

#include <armadillo>
#include "combinadics.hpp"
#include <cmath>
#include <iomanip>

enum basis_type
{
	full,
	const_particles
};

template <typename T>
class hamiltonian
{
protected:
	uint64_t n_sites = 0;
	uint64_t n_particles = 0;
	uint64_t n_states = 0;
	std::string name = "base class";
	basis_type type = full;
	combinadics *basis = nullptr;
	bool diagonalized = false;
	bool constructed = false;

	T ham;
	T eigenvectors;
	arma::vec eigenvalues;

public:
	// hamiltonian();
	~hamiltonian()
	{
		delete basis;
		// delete ham;
		// delete eigenvectors;
		// delete eigenvalues;

		basis = nullptr;
		// ham = nullptr;
		// eigenvectors = nullptr;
		// eigenvalues = nullptr;
		// std::cout<<"Hamiltonian deleted." << std::endl;
	}
	combinadics *get_basis() const { return basis; }
	void set_basis(combinadics *new_basis) { basis = new_basis; }

	uint64_t get_n_sites() const { return n_sites; }
	void set_n_sites(u_int64_t new_n_sites) { n_sites = new_n_sites; }

	uint64_t get_n_particles() const { return n_particles; }
	void set_n_particles(uint64_t new_n_particles) { n_particles = new_n_particles; }

	uint64_t get_n_states() const { return n_states; }
	void set_n_states(uint64_t new_n_states) { n_states = new_n_states; }

	std::string get_name() { return name; }
	void set_name(std::string new_name) { name = new_name; }

	basis_type get_basis_type() { return type; }
	void set_basis_type(basis_type new_type) { type = new_type; }

	bool is_diag() { return diagonalized; }

	T *get_ham() { return &ham; }
	void set_ham(T *new_ham) { &ham = new_ham; }

	T *get_eigenvectors()
	{
		if (diagonalized)
			return &eigenvectors;
		else
		{
			diagonalize();
			return &eigenvectors;
		}
	}

	void set_eigenvectors(T *new_eigenvectors) { &eigenvectors = new_eigenvectors; }

	arma::vec *get_eigenvalues()
	{

		if (diagonalized)
			return &eigenvalues;
		else
		{
			diagonalize();
			return &eigenvalues;
		}
	}

	void set_eigenvalues(arma::vec *new_eigenvalues) { &eigenvalues = new_eigenvalues; }

	int diagonalize()
	{
		if(!constructed) regenerate();

		try
		{
			if (ham.n_cols != n_states)
				throw "hamiltonian";
			// if(eigenvectors. == nullptr) throw "eigenvectors matrix";
			// if(eigenvalues. == nullptr) throw "eigenvalues vector"
		}
		catch (const std::string &e)
		{
			std::cerr << e << " is not/wrongly initialized. Aborting...\n";
			exit(EXIT_FAILURE);
		}

		arma::eig_sym(eigenvalues, eigenvectors, ham);
		diagonalized = true;

		return 0;
	}

	binary_rep get_state_binary(uint64_t idx)
	{
		try
		{
			if (idx > n_states)
				throw idx;
		}
		catch (uint64_t idx)
		{
			std::cerr << "Index " << idx << " in get_state_binary(uint64_t idx) is larger than number of states (" << n_states << ")\n";
			std::cerr << "Aborting..." << std::endl;
			exit(EXIT_FAILURE);
		}

		switch (type)
		{
		case full:
		{

			binary_rep res;
			res.sites = n_sites;
			res.state = new bool[n_sites];
			basis->fast_d2b(idx, res.state);
			return res;
			break;
		}
		case const_particles:
		{
			return basis->get_state_binary(idx);
			break;
		}

		default:
			std::cerr << "No such basis type (get_state_binary). Aborting...\n";
			exit(EXIT_FAILURE);
			break;
		}
	}
	uint64_t reverse_index(uint64_t state)
	{
		switch (type)
		{
		case full:
		{
			return state;
			break;
		}
		case const_particles:
		{
			return basis->reverse_index(state);
			break;
		}
		default:
			std::cerr << "No such basis type (reverse_index). Aborting...\n";
			exit(EXIT_FAILURE);
			break;
		}
	}
	void print_eigenvalues()
	{
		for (auto val : eigenvalues)
			std::cout << std::setprecision(17) << val << std::endl;
	}
	void print_eigenvectors() { std::cout << std::setprecision(5) << eigenvectors << std::endl; }
	void print_hamiltonian() { std::cout << std::setprecision(5) << ham << std::endl; }

	double get_eigenvalue(uint64_t idx)
	{

		if (diagonalized)
			return eigenvalues(idx);
		else
		{
			diagonalize();
			return eigenvalues(idx);
		}
	}
	void clear_ham_matrix()
	{
		ham.clear();
		// diagonalized = false;
		constructed = false;
	}
	void clear_eigenvectors()
	{
		eigenvectors.clear();
		diagonalized = false;
	}
	void clear_eigenvalues()
	{
		eigenvalues.clear();
		diagonalized = false;
	}

	virtual void regenerate()
	{
		std::cerr <<"This is base regenerate function. You should not be here.";
		exit(EXIT_FAILURE);
	}
};
#endif