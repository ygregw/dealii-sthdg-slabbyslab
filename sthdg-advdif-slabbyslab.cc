#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{ using namespace dealii::LinearAlgebraPETSc; }

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <iostream>
#include <sys/resource.h>
#include <unistd.h>

//#define SemiUpwinding
#define SemiCentreFlux

long get_mem_usage(){
	struct rusage myusage;
	getrusage(RUSAGE_SELF, &myusage);
	return myusage.ru_maxrss;
}

namespace SpaceTimeAdvecDiffuIPH
{
	using namespace dealii;
	const double pi = numbers::PI;

	template <int dim>
		class Solution : public Function<dim> {
			public:
				Solution (const double nu) : Function<dim>(), nu(nu) {}
				double value (const Point<dim> &p, const unsigned int component = 0) const override;
				Tensor<1, dim> gradient (const Point<dim> &p,
						const unsigned int /*component*/ = 0) const override;
			private:
				const double nu;
		};

	template <int dim>
		double Solution<dim>::value (const Point<dim> &p, const unsigned int) const {
			double return_value = 0;
			const double t = p[0];
			const double x = p[1];
			const double y = p[2];
			const double x1 = x*std::cos(4*t)+y*std::sin(4*t);
			const double x2 = -x*std::sin(4*t)+y*std::cos(4*t);
			const double x1c = -0.2;
			const double x2c = 0.1;
			const double sigma_sq = 0.01; // sigma * sigma
			const double x1d_sq = (x1-x1c)*(x1-x1c);
			const double x2d_sq = (x2-x2c)*(x2-x2c);
			switch (dim)
			{
				case 3:
					return_value =
						std::exp(-(x1d_sq+x2d_sq)/(2*sigma_sq+4*nu*t))
						* sigma_sq/(sigma_sq+2*nu*t);
					break;
				default:
					Assert (false, ExcNotImplemented());
			}
			return return_value;
		};

	template <int dim>
		Tensor<1,dim> Solution<dim>::gradient (const Point<dim> &p, const unsigned int) const {
			Tensor<1,dim> return_value;
			const double t = p[0];
			const double x = p[1];
			const double y = p[2];
			const double x1 = x*std::cos(4*t)+y*std::sin(4*t);
			const double x2 = -x*std::sin(4*t)+y*std::cos(4*t);
			const double x1c = -0.2;
			const double x2c = 0.1;
			const double x1d = x1-x1c;
			const double x2d = x2-x2c;
			const double xd_sq = x1d*x1d + x2d*x2d;
			const double sigma_sq = 0.01; // sigma * sigma
			const double sigma_nut = sigma_sq + 2*nu*t;
			const double exp_term = std::exp(-xd_sq/(2*sigma_nut));
			switch (dim)
			{
				case 3:
						return_value[0] =
							sigma_sq/(sigma_nut*sigma_nut) * exp_term *
							( -2*nu + nu*xd_sq/(sigma_nut) - 4*(x1d*x2-x2d*x1));
						return_value[1] =
							- sigma_sq/(sigma_nut*sigma_nut) * exp_term *
							(x1d*std::cos(4*t)-x2d*std::sin(4*t));
						return_value[2] =
							- sigma_sq/(sigma_nut*sigma_nut) * exp_term *
							(x1d*std::sin(4*t)+x2d*std::cos(4*t));
					break;
				default:
					Assert (false, ExcNotImplemented());
			}
			return return_value;
		};

	template <int dim>
		class AdvectionVelocity : public TensorFunction<1,dim> {
			public:
				AdvectionVelocity() : TensorFunction<1,dim>() {}
				Tensor<1,dim> value (const Point<dim> &p) const override;
		};

	template <int dim>
		Tensor<1,dim> AdvectionVelocity<dim>::value(const Point<dim> &p) const {
			Tensor<1,dim> advection;
			const double xx = p[1];
			const double yy = p[2];
			switch (dim)
			{
				case 3:
					advection[0] = 1;
					advection[1] = -4*yy;
					advection[2] = 4*xx;
					break;
				default:
					Assert(false, ExcNotImplemented());
			}
			return advection;
		};

	template <int dim>
		class RightHandSide : public Function<dim> {
		public:
			RightHandSide (const double nu) : Function<dim>(), nu(nu) {}
			double value (const Point<dim> &p, const unsigned int component = 0) const override;
		private:
			const AdvectionVelocity<dim> advection_velocity;
			const double nu;
		};

	template <int dim>
		double RightHandSide<dim>::value (const Point<dim> &p, const unsigned int) const {
			(void)p;
			double return_value = 0;
			switch (dim)
			{
				case 3:
					return_value = 0;
					break;
				default:
					Assert (false, ExcNotImplemented());
			}
			return return_value;
		};


	template <int dim>
		class SpaceTimeHDG
		{
			public:
				SpaceTimeHDG(const unsigned int degree, const double nu, const unsigned int num_cycle, const bool ifvtk);
				void run();

			private:
				void deform_slab(const double delta_t, const double current_time);
				void setup_system();
				void assemble_system(const bool reconstruct_trace = false, const bool first_time_step = true);
        void solve();
        void new_initial_cond();
				void calculate_errors();
				void output_results(const unsigned int cycle, const unsigned int time_step);

				MPI_Comm mpi_communicator;

				parallel::distributed::Triangulation<dim> triangulation;

				std::vector<Point<dim>>  init_coordinates;

				// local element solution
				FE_DGQ<dim>			fe_local;
				DoFHandler<dim> dof_handler_local;
				LA::MPI::Vector locally_owned_solution_local;
				IndexSet locally_owned_dofs_local;

				// global facet solution
				FE_FaceQ<dim>   fe;
				DoFHandler<dim> dof_handler;
				LA::MPI::Vector locally_relevant_solution;
				IndexSet locally_owned_dofs;
				IndexSet locally_relevant_dofs;

        std::vector<std::vector<double>> new_initial_values;

				AffineConstraints<double> constraints;

				LA::MPI::SparseMatrix system_matrix;
				LA::MPI::Vector       system_rhs;

				ConvergenceTable convergence_table;

				const double nu; // diffusion parameter
				const unsigned int num_cycle; // num of refinement levels

        // l2, sh1, th1, adv_facet, dif_facet, neumann, supg, total
        std::vector<double> error_list;

				ConditionalOStream pcout;
				TimerOutput computing_timer;

				const bool ifvtk;
		};


	template <int dim>
		SpaceTimeHDG<dim>::SpaceTimeHDG(const unsigned int degree,
				const double nu,
				const unsigned int num_cycle,
				const bool ifvtk
				) :
			mpi_communicator(MPI_COMM_WORLD),
			triangulation(mpi_communicator),
			fe_local(degree),
			dof_handler_local(triangulation),
			fe(degree),
			dof_handler(triangulation),
			nu(nu),
			num_cycle(num_cycle),
			error_list(8, 0.0),
			pcout(std::cout,
					(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
			computing_timer(mpi_communicator,
					pcout,
					TimerOutput::never,
					TimerOutput::wall_times),
			ifvtk(ifvtk)
	{}

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // DEFORM_SLAB
  // 1. takes a current_time value;
  // 2. re-generate the initial coarse slab;
	// 3. mark the ring for refinement later;
	// 4. deform vertices;
  // 5. execute refinment and reinit dof handlers;
	// 6. mark boundary face ids.
	template <int dim>
		void SpaceTimeHDG<dim>::deform_slab(const double delta_t, const double current_time)
		{
			triangulation.clear();
			TimerOutput::Scope t(computing_timer, "deform_slab");
			const double L = 0.5;
			const Point<dim,double> p1(0, -L, -L);
			const Point<dim,double> p2(delta_t, L, L);
			unsigned int num_elem_s = floor(2*L/delta_t);
			const std::vector<unsigned int> repetitions = {1, num_elem_s, num_elem_s};
			GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

			const double A = 0.1;

			// making hanging nodes around pulse orbit.
			for (auto &cell : triangulation.active_cell_iterators()) {
				if (cell->is_locally_owned())
				{
					const Point<dim> cell_center = cell->center();
					const double dif_pulse_center = std::fabs(
							std::sqrt(cell_center[1]*cell_center[1] + cell_center[2]*cell_center[2])-0.2
							);
					if (dif_pulse_center < 0.1) {
						cell->set_refine_flag(RefinementCase<dim>::cut_xyz);
					}
				}
			}

			std::set<unsigned int> deformed_v;
			std::set<unsigned int>::iterator found;
			for (auto &cell : triangulation.active_cell_iterators())
				{
					for (const auto i : cell->vertex_indices())
					{
						unsigned int vi = cell->vertex_index(i);
						found = deformed_v.find(vi);
						if (found != deformed_v.end())
							continue;
						else {
							Point<dim> &v = cell->vertex(i);
							double vt = v[0] + current_time;
							double vx = v[1];
							double vy = v[2];
							for (unsigned int i =0; i < dim; ++i)
							{
								switch (i) {
									case 0:
										v[i] = vt;
										break;
									case 1:
										v[i] = vx + A*(L - vx)*sin(2*pi*(L - vy + vt));
										break;
									case 2:
										v[i] = vy + A*(L - vy)*sin(2*pi*(L - vx + vt));
										break;
								}
							}
							deformed_v.insert(vi);
						}
					}
				}

			triangulation.execute_coarsening_and_refinement();
			dof_handler_local.reinit(triangulation);
			dof_handler.reinit(triangulation);
			for (auto &face : triangulation.active_face_iterators())
				if (face->at_boundary())
				{
					const Point<dim> face_center = face->center();
					// top R-boundary has id 1;
					if (std::fabs(face_center[0]-current_time) < 1e-12) {
						face->set_boundary_id(1);
					}
					// bottom R-boundary has id 2;
					else if (std::fabs(face_center[0]-current_time-delta_t) < 1e-12) {
						face->set_boundary_id(2);
					} else {
						// Q-boundary has id 0;
						face->set_boundary_id(0);
					}
				}

		}

	template <int dim>
		void SpaceTimeHDG<dim>::setup_system()
		{
			TimerOutput::Scope t(computing_timer, "setup_system");

			dof_handler_local.distribute_dofs(fe_local);
			locally_owned_dofs_local = dof_handler_local.locally_owned_dofs();

			dof_handler.distribute_dofs(fe);
			locally_owned_dofs = dof_handler.locally_owned_dofs();
			locally_relevant_dofs =
				DoFTools::extract_locally_relevant_dofs(dof_handler);

			locally_owned_solution_local.reinit(locally_owned_dofs_local,
					mpi_communicator);
			locally_relevant_solution.reinit(locally_owned_dofs,
					locally_relevant_dofs,
					mpi_communicator);

			system_rhs.reinit(locally_owned_dofs, mpi_communicator);

			// Assign Dirichlet boundary values from the pre-defined Solution class
			// Dirichlet boundary: Q-boundary
			// Neumann boundary: bottom R-boundary (simply u = g)
			constraints.clear();
			constraints.reinit(locally_relevant_dofs);
			DoFTools::make_hanging_node_constraints(dof_handler, constraints);
			std::map<types::boundary_id, const Function<dim> *> boundary_functions;
			Solution<dim> solution_function(nu);
			boundary_functions[0] = &solution_function;
			VectorTools::interpolate_boundary_values(dof_handler,
					boundary_functions,
					constraints);
			constraints.close();

			DynamicSparsityPattern dsp(locally_relevant_dofs);
			DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
			SparsityTools::distribute_sparsity_pattern(dsp,
					dof_handler.locally_owned_dofs(),
					mpi_communicator,
					locally_relevant_dofs);
			system_matrix.reinit(locally_owned_dofs,
					locally_owned_dofs,
					dsp,
					mpi_communicator);
		}

	template <int dim>
		void SpaceTimeHDG<dim>::assemble_system(const bool trace_reconstruct, const bool first_time_step)
		{
			TimerOutput::Scope t(computing_timer, "assemble_system");
			const QGauss<dim> quadrature_formula(fe.degree+1);
			const QGauss<dim-1> face_quadrature_formula(fe.degree+1);

			const UpdateFlags local_flags(update_values |
					update_gradients |
					update_JxW_values |
					update_quadrature_points);

			const UpdateFlags local_face_flags(update_values |
					update_gradients |
					update_JxW_values);

			const UpdateFlags flags(update_values |
					update_normal_vectors |
					update_quadrature_points |
					update_JxW_values);

			FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
			Vector<double> cell_vector(fe.dofs_per_cell);
			std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

			FEValues<dim> fe_values_local(fe_local,quadrature_formula,local_flags);
			FEFaceValues<dim> fe_face_values_local(fe_local,face_quadrature_formula,local_face_flags);
			FEFaceValues<dim> fe_face_values(fe,face_quadrature_formula,flags);

#if defined(SemiUpwinding) || defined(SemiCentreFlux)
			const QGauss<dim-1> face_quadrature_formula_betamax(2);
			const UpdateFlags flags_betamax(update_normal_vectors | update_quadrature_points);
			FEFaceValues<dim> fe_face_values_betamax(fe,face_quadrature_formula_betamax,flags_betamax);
			const unsigned int n_face_q_points_betamax =
				fe_face_values_betamax.get_quadrature().size();
#endif

			AdvectionVelocity<dim> advection_velocity;
			RightHandSide<dim> right_hand_side(nu);
			const Solution<dim> exact_solution(nu);

			const unsigned int n_q_points =
				fe_values_local.get_quadrature().size();
			const unsigned int n_face_q_points =
				fe_face_values_local.get_quadrature().size();
			const unsigned int loc_dofs_per_cell =
				fe_values_local.get_fe().n_dofs_per_cell();

			FullMatrix<double> ll_matrix(fe_local.dofs_per_cell,fe_local.dofs_per_cell);
			FullMatrix<double> lf_matrix(fe_local.dofs_per_cell,fe.dofs_per_cell);
			FullMatrix<double> fl_matrix(fe.dofs_per_cell,fe_local.dofs_per_cell);
			FullMatrix<double> tmp_matrix(fe.dofs_per_cell,fe_local.dofs_per_cell);
			Vector<double>     l_rhs(fe_local.dofs_per_cell);
			Vector<double>     tmp_rhs(fe_local.dofs_per_cell);

			std::vector<double>						u_phi(fe_local.dofs_per_cell);
			std::vector<Tensor<1, dim>>		u_phi_grad(fe_local.dofs_per_cell);
			std::vector<Tensor<1, dim-1>> u_phi_grad_s(fe_local.dofs_per_cell);
			std::vector<double>						tr_phi(fe.dofs_per_cell);
			std::vector<double>						trace_values(face_quadrature_formula.size());
			std::vector<double>						earlier_face_values(face_quadrature_formula.size());

			std::vector<std::vector<unsigned int>> fe_local_support_on_face(GeometryInfo<dim>::faces_per_cell);
			std::vector<std::vector<unsigned int>> fe_support_on_face(GeometryInfo<dim>::faces_per_cell);
			{
				for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
					for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
					{
						if (fe.has_support_on_face(i,face))
							fe_support_on_face[face].push_back(i);
					}
			}

			typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
			for(; cell!=endc; ++cell)
				if (cell->is_locally_owned())
				{
					typename DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_local);

					ll_matrix = 0;
					l_rhs = 0;
					if (!trace_reconstruct)
					{
						lf_matrix = 0;
						fl_matrix = 0;
						cell_matrix = 0;
						cell_vector = 0;
					}
					fe_values_local.reinit(loc_cell);

					for (unsigned int q = 0; q < n_q_points; ++q)
					{
						const double rhs_value = right_hand_side.value(fe_values_local.quadrature_point(q));
						const Tensor<1, dim> advection = advection_velocity.value(fe_values_local.quadrature_point(q));
						const double JxW = fe_values_local.JxW(q);
						for (unsigned int k = 0; k < loc_dofs_per_cell; ++k)
						{
							u_phi[k] = fe_values_local.shape_value(k, q);
							u_phi_grad[k] = fe_values_local.shape_grad(k, q);
							for (unsigned int m = 0; m < dim-1; ++m)
								u_phi_grad_s[k][m] = u_phi_grad[k][m+1];
						}
						for (unsigned int i = 0; i < loc_dofs_per_cell; ++i)
						{
							for (unsigned int j = 0; j < loc_dofs_per_cell; ++j)
								ll_matrix(i, j) += (
										nu * u_phi_grad_s[i] * u_phi_grad_s[j] -
										(u_phi_grad[i] * advection) * u_phi[j]
										) * JxW;
							l_rhs(i) += u_phi[i] * rhs_value * JxW;
						}
					}

					for (const auto face_no : cell->face_indices())
					{
						fe_face_values_local.reinit(loc_cell, face_no);
						fe_face_values.reinit(cell, face_no);
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
						fe_face_values_betamax.reinit(cell, face_no);
						std::vector<double>	beta_flux(n_face_q_points_betamax);
						for (unsigned int q = 0; q < n_face_q_points_betamax; ++q) {
							const Point<dim> quadrature_point_betamax = fe_face_values_betamax.quadrature_point(q);
							const Tensor<1, dim> normal_betamax= fe_face_values_betamax.normal_vector(q);
							const Tensor<1, dim> advection_betamax = advection_velocity.value(quadrature_point_betamax);
							beta_flux[q] = normal_betamax * advection_betamax;
						}
						double beta_s = 0;
#if defined(SemiUpwinding)
						double beta_flux_max = beta_flux[0];
						for (unsigned int i = 1; i < beta_flux.size(); ++i) {
							if (beta_flux[i] > beta_flux_max)
								beta_flux_max = beta_flux[i];
						}
						beta_s = std::max(beta_flux_max, 0);
#else
						double abs_beta_flux_max = std::fabs(beta_flux[0]);
						for (unsigned int i = 1; i < beta_flux.size(); ++i) {
							if (std::fabs(beta_flux[i]) > abs_beta_flux_max)
								abs_beta_flux_max = std::fabs(beta_flux[i]);
						}
						beta_s = abs_beta_flux_max;
#endif
#endif
						if (trace_reconstruct)
							fe_face_values.get_function_values(locally_relevant_solution, trace_values);

						for (unsigned int q = 0; q < n_face_q_points; ++q)
						{
							const double JxW = fe_face_values.JxW(q);
							const Point<dim> quadrature_point = fe_face_values.quadrature_point(q);
							const Tensor<1, dim> normal = fe_face_values.normal_vector(q);
							Tensor<1, dim> normal_s = normal;
							normal_s[0]=0;
							const Tensor<1, dim> advection = advection_velocity.value(quadrature_point);

							const double hK = cell->measure() / cell->face(face_no)->measure();
							const double alpha = 8.0 * fe.degree * fe.degree;
							const double ip_stab = nu*alpha/hK;
							const double beta_n = advection * normal;
							const double beta_n_abs = std::abs(beta_n);

							for (unsigned int k = 0; k < fe_local.dofs_per_cell; ++k)
							{
								u_phi[k] = fe_face_values_local.shape_value(k, q);
								u_phi_grad[k] = fe_face_values_local.shape_grad(k,q);
							}

							// global system D-CA^{-1}B\G-CA^{-1}F
							if (!trace_reconstruct)
							{
								for (unsigned int k = 0; k < fe_support_on_face[face_no].size(); ++k)
									tr_phi[k] = fe_face_values.shape_value(fe_support_on_face[face_no][k], q);

								for (unsigned int i = 0; i < fe_local.dofs_per_cell; ++i)
									for (unsigned int j = 0; j < fe_support_on_face[face_no].size(); ++j)
									{
										const unsigned int jj = fe_support_on_face[face_no][j];
										// if R-faces else Q-faces
										if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											lf_matrix(i, jj) +=
												(beta_n - beta_s) * u_phi[i] * tr_phi[j] * JxW;
											fl_matrix(jj, i) -=
												- beta_s * u_phi[i] * tr_phi[j] * JxW;
#else
											lf_matrix(i, jj) +=
												0.5*(beta_n - beta_n_abs) * u_phi[i] * tr_phi[j] * JxW;
											fl_matrix(jj, i) -=
												-0.5*(beta_n + beta_n_abs) * u_phi[i] * tr_phi[j] * JxW;
#endif
										}
										else {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											lf_matrix(i, jj) +=
												((nu * u_phi_grad[i] * normal_s +
													((beta_n - beta_s) - ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;

											fl_matrix(jj, i) -=
												(( nu * u_phi_grad[i] * normal_s -
													 (beta_s + ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;
#else
											lf_matrix(i, jj) +=
												((nu * u_phi_grad[i] * normal_s +
													(0.5*(beta_n - beta_n_abs) - ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;

											fl_matrix(jj, i) -=
												(( nu * u_phi_grad[i] * normal_s -
													 (0.5*(beta_n + beta_n_abs) + ip_stab) * u_phi[i]
												 ) * tr_phi[j]) * JxW;
#endif
										}
									}

								for (unsigned int i = 0; i < fe_support_on_face[face_no].size(); ++i)
									for (unsigned int j = 0; j < fe_support_on_face[face_no].size(); ++j)
									{
										const unsigned int ii = fe_support_on_face[face_no][i];
										const unsigned int jj = fe_support_on_face[face_no][j];
										// if R-faces else Q-faces
										if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											cell_matrix(ii, jj) +=
												- (beta_n - beta_s) * tr_phi[i] * tr_phi[j] * JxW;
#else
											cell_matrix(ii, jj) +=
												- 0.5*(beta_n - beta_n_abs) * tr_phi[i] * tr_phi[j] * JxW;
#endif
										}
										else {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
											cell_matrix(ii, jj) +=
												( (-(beta_n - beta_s) + ip_stab)
													* tr_phi[i] * tr_phi[j]) * JxW;
#else
											cell_matrix(ii, jj) +=
												( (-0.5*(beta_n - beta_n_abs) + ip_stab)
													* tr_phi[i] * tr_phi[j]) * JxW;
#endif
										}
									}

								// Neumann boundary condition

								if (cell->face(face_no)->at_boundary() && (cell->face(face_no)->boundary_id() != 0))
								{
									for (unsigned int i = 0; i < fe_support_on_face[face_no].size(); ++i)
										for (unsigned int j = 0; j < fe_support_on_face[face_no].size(); ++j)
										{
											const unsigned int ii = fe_support_on_face[face_no][i];
											const unsigned int jj = fe_support_on_face[face_no][j];
											cell_matrix(ii, jj) +=
												( 0.5 * (beta_n + beta_n_abs) * tr_phi[i] * tr_phi[j]) * JxW;
										}

									if (cell->face(face_no)->boundary_id() == 1) {
										if (first_time_step) {
											const double neumann_value = -exact_solution.gradient(quadrature_point) * normal_s;
											const double dirichlet_value = exact_solution.value(quadrature_point);
											for (unsigned int i = 0; i < fe_support_on_face[face_no].size(); ++i)
											{
												const unsigned int ii = fe_support_on_face[face_no][i];
												cell_vector(ii) += -tr_phi[i] * (
														nu * neumann_value + 0.5 * (beta_n - beta_n_abs) * dirichlet_value
														) * JxW;
											}
										}
										else {
											// if not the first space-time slab
											const double neumann_value = -exact_solution.gradient(quadrature_point) * normal_s;
											const double dirichlet_value = new_initial_values[cell->active_cell_index()][q];
											for (unsigned int i = 0; i < fe_support_on_face[face_no].size(); ++i)
											{
												const unsigned int ii = fe_support_on_face[face_no][i];
												cell_vector(ii) += -tr_phi[i] * (
														nu * neumann_value + 0.5 * (beta_n - beta_n_abs) * dirichlet_value
														) * JxW;
											}
										}
									}

								}

							}

							for (unsigned int i = 0; i < fe_local.dofs_per_cell; ++i)
								for (unsigned int j = 0; j < fe_local.dofs_per_cell; ++j)
								{
									// if R-faces else Q-faces
									if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										ll_matrix(i, j) += (beta_s) * u_phi[i] * u_phi[j] * JxW;
#else
										ll_matrix(i, j) += 0.5 * (beta_n + beta_n_abs) * u_phi[i] * u_phi[j] * JxW;
#endif
									}
									else{
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										ll_matrix(i, j) += (
												ip_stab * u_phi[i] * u_phi[j]
												- nu * u_phi_grad[i] * normal_s * u_phi[j]
												- nu * u_phi[i] * u_phi_grad[j] * normal_s
												+ (beta_s) * u_phi[i] * u_phi[j]
												) * JxW;
#else
										ll_matrix(i, j) += (
												ip_stab * u_phi[i] * u_phi[j]
												- nu * u_phi_grad[i] * normal_s * u_phi[j]
												- nu * u_phi[i] * u_phi_grad[j] * normal_s
												+ 0.5 * (beta_n + beta_n_abs) * u_phi[i] * u_phi[j]
												) * JxW;
#endif
									}
								}

							//AU = F-B{U_hat}, here is the -B{U_hat} part.
							//{U_hat} => trace_values
							if (trace_reconstruct)
								for (unsigned int i = 0; i < fe_local.dofs_per_cell; ++i)
								{
									// tr_phi in lf_matrix replaced by trace_values
									//
									// if R-faces else Q-faces
									if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12)) {
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										l_rhs(i) -=  (beta_n - beta_s) * u_phi[i] * trace_values[q] * JxW;
#else
										l_rhs(i) -=  0.5*(beta_n - beta_n_abs) * u_phi[i] * trace_values[q] * JxW;
#endif
									}
									else{
#if defined(SemiUpwinding) || defined(SemiCentreFlux)
										l_rhs(i) -= (
												nu * u_phi_grad[i] * normal_s
												+ ((beta_n - beta_s) - ip_stab) * u_phi[i]
												) * trace_values[q] * JxW;
#else
										l_rhs(i) -= (
												nu * u_phi_grad[i] * normal_s
												+ (0.5*(beta_n - beta_n_abs) - ip_stab) * u_phi[i]
												) * trace_values[q] * JxW;
#endif
									}
								}
						}
					}

					ll_matrix.gauss_jordan();


					if (trace_reconstruct == false)
					{
						// tmp_mat = fl_mat * ll_mat^{-1}
						//         = -CA^{-1}
						fl_matrix.mmult(tmp_matrix, ll_matrix);
						// cel_vec = tmp_mat * l_rhs
						//         = -CA^{-1}F
						tmp_matrix.vmult_add(cell_vector, l_rhs);
						// cel_mat += tmp_mat * lf_mat
						//				 = (D)-CA^{-1}B
						tmp_matrix.mmult(cell_matrix, lf_matrix, true);
						cell->get_dof_indices(local_dof_indices);
					}
					else
					{
						// tmp_rhs = ll_mat * l_rhs
						//				 = A^{-1}(F-BV)
						ll_matrix.vmult(tmp_rhs, l_rhs);
						loc_cell->set_dof_values(tmp_rhs, locally_owned_solution_local);
					}

					// copy local to global
					if (trace_reconstruct == false)
					{
						constraints.distribute_local_to_global(cell_matrix,
								cell_vector,
								local_dof_indices,
								system_matrix,
								system_rhs);
					}

				}

			if (trace_reconstruct == false)
			{
				system_matrix.compress(VectorOperation::add);
				system_rhs.compress(VectorOperation::add);
			}
		}

	template <int dim>
		void SpaceTimeHDG<dim>::solve()
    {
			TimerOutput::Scope t(computing_timer, "solve");

			LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
					mpi_communicator);
			SolverControl solver_control(dof_handler.n_dofs(), 1e-12);

			LA::SolverGMRES::AdditionalData addata_gmres(300, false);
			LA::SolverGMRES solver(solver_control, mpi_communicator, addata_gmres);

			//%%%%%%%% SETTINGS FOR PETSC AMG %%%%%%%%%%%%%%%%%%%
			 LA::MPI::PreconditionAMG preconditioner;
			 LA::MPI::PreconditionAMG::AdditionalData data;
			 data.symmetric_operator = false;
			 preconditioner.initialize(system_matrix, data);
			//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


			solver.solve(system_matrix,
					completely_distributed_solution,
					system_rhs,
					preconditioner);

			constraints.distribute(completely_distributed_solution);

			locally_relevant_solution = completely_distributed_solution;

			assemble_system(true,true);// second arg doesn't matter
																																		 // when first is true
		}

	template <int dim>
		void SpaceTimeHDG<dim>::new_initial_cond()
    {
			TimerOutput::Scope t(computing_timer, "new_init_cond");
			// This function is implemented assuming there are at most
			// two local time steps in a single slab;
			// (1) In the active cell loop below, we extract facet
			// solution on the top boundary (boundary_id=2) and
			// if the bottom face of the same cell is part of the
			// slab boundary, we associate the extracted solution to
			// its active cell index;
			// the other scenario is assumed to be that the bottom face
			// of the cell is an interior face (say F_int). In this
			// case, we associate the extracted solution to the active
			// cell index of neighbor cell on the other side of F_int.
			// (2) We shouldn't allow more steps than that anyways;
			const QGauss<dim> quadrature_formula(fe.degree+1);
			const QGauss<dim-1> face_quadrature_formula(fe.degree+1);

			const UpdateFlags flags(update_values |
					update_normal_vectors |
					update_quadrature_points |
					update_JxW_values);

			FEFaceValues<dim> fe_face_values(fe,face_quadrature_formula,flags);

			const unsigned int n_face_q_points =
				fe_face_values.get_quadrature().size();

      // store new initial conditions
      new_initial_values.resize(triangulation.n_active_cells());
      for(unsigned int i = 0; i < new_initial_values.size(); ++i) {
        new_initial_values[i].resize(n_face_q_points);
        std::fill(new_initial_values[i].begin(), new_initial_values[i].end(), 0.0);
      }
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
      for(; cell!=endc; ++cell)
				if (cell->is_locally_owned()) {
					for (const auto face_no : cell->face_indices()) {
						if (cell->face(face_no)->at_boundary() && cell->face(face_no)->boundary_id() == 2) {
							fe_face_values.reinit(cell, face_no);
							std::vector<double> temp_new_initial_values;
							temp_new_initial_values.resize(n_face_q_points);
							fe_face_values.get_function_values(locally_relevant_solution, temp_new_initial_values);
							for (const auto opposite_face_no : cell->face_indices()) {
								if (opposite_face_no != face_no) {
									fe_face_values.reinit(cell, opposite_face_no);
									if (std::fabs(fe_face_values.normal_vector(0)[0] + 1) < 1e-10) {
										if (cell->face(opposite_face_no)->at_boundary()) {
											new_initial_values[cell->active_cell_index()] = temp_new_initial_values;
											break;
										} else {
											new_initial_values[cell->neighbor(opposite_face_no)->active_cell_index()] = temp_new_initial_values;
											break;
										}
									}
								}
							}
						}
					}
				}
    }



	template <int dim>
		void SpaceTimeHDG<dim>::calculate_errors()
		{
			Vector<float> difference_per_cell_l2(triangulation.n_active_cells());
      Vector<float> difference_per_cell_sh1(triangulation.n_active_cells());
      Vector<float> difference_per_cell_th1(triangulation.n_active_cells());
			Vector<float> difference_per_cell_dif_jump(triangulation.n_active_cells());
      Vector<float> difference_per_cell_adv_jump(triangulation.n_active_cells());
      Vector<float> difference_per_cell_neumann(triangulation.n_active_cells());
      Vector<float> difference_per_cell_supg(triangulation.n_active_cells());

			const QGauss<dim> quadrature_formula(fe.degree+2);
			const QGauss<dim-1> face_quadrature_formula(fe.degree+2);

			const UpdateFlags local_flags(update_values |
					update_gradients |
					update_JxW_values |
					update_quadrature_points);

			const UpdateFlags local_face_flags(update_values |
					update_gradients |
					update_JxW_values);

			const UpdateFlags flags(update_values |
					update_normal_vectors |
					update_quadrature_points |
					update_JxW_values);


			std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

			FEValues<dim> fe_values_local(fe_local,quadrature_formula,local_flags);
			FEFaceValues<dim> fe_face_values_local(fe_local,face_quadrature_formula,local_face_flags);
			FEFaceValues<dim> fe_face_values(fe,face_quadrature_formula,flags);

			const AdvectionVelocity<dim> advection_velocity;
			const RightHandSide<dim> right_hand_side(nu);
			const Solution<dim> exact_solution(nu);

#if defined(SemiUpwinding) || defined(SemiCentreFlux)
			const QGauss<dim-1> face_quadrature_formula_betamax(2);
			const UpdateFlags flags_betamax(update_normal_vectors | update_quadrature_points);
			FEFaceValues<dim> fe_face_values_betamax(fe,face_quadrature_formula_betamax,flags_betamax);
#endif
			const unsigned int n_q_points = fe_values_local.get_quadrature().size();
			const unsigned int n_face_q_points = fe_face_values_local.get_quadrature().size();

      double temp_err_l2 = 0;
      double temp_err_sh1 = 0;
      double temp_err_th1 = 0;
      double temp_err_dif_jump = 0;
      double temp_err_adv_jump = 0;
      double temp_err_neumann = 0;
      double temp_err_supg = 0;

			typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
			for(; cell!=endc; ++cell)
				if (cell->is_locally_owned())
				{
					typename DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_local);
					fe_values_local.reinit(loc_cell);
					const double hK = std::pow(cell->measure(),1.0/dim);

					std::vector<double> elem_sol_vals(quadrature_formula.size());
					fe_values_local.get_function_values(locally_owned_solution_local, elem_sol_vals);
					std::vector<Tensor<1,dim>> elem_sol_grads(quadrature_formula.size());
					fe_values_local.get_function_gradients(locally_owned_solution_local, elem_sol_grads);

					for (unsigned int q = 0; q < n_q_points; ++q)
					{
						const Tensor<1,dim> advection = advection_velocity.value(fe_values_local.quadrature_point(q));
						const double exact_value = exact_solution.value(fe_values_local.quadrature_point(q));
						const Tensor<1,dim> exact_grad_value = exact_solution.gradient(fe_values_local.quadrature_point(q));
						const double JxW = fe_values_local.JxW(q);
						temp_err_l2 += ((exact_value-elem_sol_vals[q])*(exact_value-elem_sol_vals[q])) * JxW;
						temp_err_supg += hK * hK *
							(advection * (exact_grad_value-elem_sol_grads[q])) *
							(advection * (exact_grad_value-elem_sol_grads[q]))
							* JxW;
						for (unsigned int c = 1; c < dim; ++c)
							temp_err_sh1 += (
									(exact_grad_value[c]-elem_sol_grads[q][c]) * (exact_grad_value[c]-elem_sol_grads[q][c])
									) * JxW;
						if (hK < nu)
							temp_err_th1 += hK * (
									(exact_grad_value[0]-elem_sol_grads[q][0]) * (exact_grad_value[0]-elem_sol_grads[q][0])
									) * JxW;
						else
							temp_err_th1 += hK * nu * (
									(exact_grad_value[0]-elem_sol_grads[q][0]) * (exact_grad_value[0]-elem_sol_grads[q][0])
									) * JxW;
					}
					difference_per_cell_l2(cell->active_cell_index())  = std::sqrt(temp_err_l2);
					difference_per_cell_sh1(cell->active_cell_index()) = std::sqrt(nu)*std::sqrt(temp_err_sh1);
					difference_per_cell_th1(cell->active_cell_index()) = std::sqrt(temp_err_th1);
					difference_per_cell_supg(cell->active_cell_index()) = std::sqrt(temp_err_supg);
					temp_err_l2  = 0;
					temp_err_sh1 = 0;
					temp_err_th1 = 0;
					temp_err_supg = 0;

					for (const auto face_no : cell->face_indices()) {
						fe_face_values_local.reinit(loc_cell, face_no);
						fe_face_values.reinit(cell, face_no);
						std::vector<double> face_sol_vals_local(face_quadrature_formula.size());
						std::vector<double> face_sol_vals(face_quadrature_formula.size());
						fe_face_values_local.get_function_values(locally_owned_solution_local, face_sol_vals_local);
						fe_face_values.get_function_values(locally_relevant_solution, face_sol_vals);

						for (unsigned int q = 0; q < n_face_q_points; ++q) {
							const double JxW = fe_face_values.JxW(q);
							const Point<dim> quadrature_point = fe_face_values.quadrature_point(q);
							const Tensor<1,dim> advection = advection_velocity.value(quadrature_point);
							const Tensor<1,dim> normal = fe_face_values.normal_vector(q);
							// if R-faces else Q-faces
							if ((std::fabs(normal[0]-(1)) < 1e-12) || (std::fabs(normal[0]+(1)) < 1e-12))
								temp_err_adv_jump +=
									(face_sol_vals[q] - face_sol_vals_local[q])*(face_sol_vals[q] - face_sol_vals_local[q]) * JxW;
							else {
								temp_err_adv_jump += std::fabs(advection * normal) *
									(face_sol_vals[q] - face_sol_vals_local[q])*(face_sol_vals[q] - face_sol_vals_local[q]) * JxW;
								temp_err_dif_jump += nu/hK *
									(face_sol_vals[q] - face_sol_vals_local[q])*(face_sol_vals[q] - face_sol_vals_local[q]) * JxW;
							}

							// Neumann boundary err term
							if (cell->face(face_no)->at_boundary() && (cell->face(face_no)->boundary_id() != 0)) {
								const double exact_value = exact_solution.value(quadrature_point);
								temp_err_neumann += std::fabs(advection * normal) *
									(face_sol_vals[q] - exact_value)*(face_sol_vals[q] - exact_value) * JxW;
							}
						}
					}
					difference_per_cell_dif_jump(cell->active_cell_index())  = std::sqrt(temp_err_dif_jump);
					difference_per_cell_adv_jump(cell->active_cell_index())  = std::sqrt(temp_err_adv_jump);
					difference_per_cell_neumann(cell->active_cell_index())  = std::sqrt(temp_err_neumann);
					temp_err_dif_jump = 0;
					temp_err_adv_jump = 0;
					temp_err_neumann = 0;
				}

			const double l2_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_l2, VectorTools::L2_norm);
			const double sH1_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_sh1, VectorTools::L2_norm);
			const double tH1_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_th1, VectorTools::L2_norm);
			const double dif_jump_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_dif_jump, VectorTools::L2_norm);
			const double adv_jump_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_adv_jump, VectorTools::L2_norm);
			const double neumann_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_neumann, VectorTools::L2_norm);
			const double supg_error =
				VectorTools::compute_global_error(triangulation,
						difference_per_cell_supg, VectorTools::L2_norm);
      error_list[0] += (l2_error*l2_error);
      error_list[1] += (sH1_error*sH1_error);
      error_list[2] += (tH1_error*tH1_error);
      error_list[3] += (dif_jump_error*dif_jump_error);
      error_list[4] += (adv_jump_error*adv_jump_error);
      error_list[5] += (neumann_error*neumann_error);
      error_list[6] += (supg_error*supg_error);
		}


	template <int dim>
		void SpaceTimeHDG<dim>::output_results(const unsigned int cycle, const unsigned int time_step)
		{
			std::string filename;
			filename = "solution";
			filename += "-q" + Utilities::int_to_string(fe.degree, 1);
			filename += "-l" + Utilities::int_to_string(cycle, 1);
			filename += "-t";

			DataOut<dim> data_out;
			std::vector<std::string> name(1, "solution");
			std::vector<DataComponentInterpretation::DataComponentInterpretation>
				comp_type(1, DataComponentInterpretation::component_is_scalar);
			data_out.add_data_vector(dof_handler_local,
					locally_owned_solution_local,
					name,
					comp_type);

			Vector<float> subdomain(triangulation.n_active_cells());
			for (unsigned int i = 0; i < subdomain.size(); ++i)
				subdomain(i) = triangulation.locally_owned_subdomain();
			data_out.add_data_vector(subdomain, "subdomain");

			data_out.build_patches(fe.degree);
			data_out.write_vtu_with_pvtu_record(
					"./vtus/", filename, time_step, mpi_communicator, 4, 8);
		}


	template <int dim>
		void SpaceTimeHDG<dim>::run()
		{
			const unsigned int n_ranks =
				Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
			std::string DAT_header = "START DATE: " + Utilities::System::get_date() +
				", TIME: " + Utilities::System::get_time();
			std::string PRB_header = "Rotating Guassian Pulse Problem, nu = ";
			std::string MPI_header = "Running with " + std::to_string(n_ranks) +
				" MPI process" + (n_ranks > 1 ? "es" : "") + ", PETSC";
			std::string SOL_header = "Finite element space: " + fe.get_name() + ", " + fe_local.get_name();
#if defined(SemiUpwinding)
			std::string MET_header = "Space-time IP-HDG, with semi-upwinding penalty";
#elif defined(SemiCentreFlux)
			std::string MET_header = "Space-time IP-HDG, with semi-centered-flux penalty";
#else
			std::string MET_header = "Space-time IP-HDG, with classic upwinding penalty";
#endif

      pcout << std::string(80, '=') << std::endl;
      pcout << DAT_header << std::endl;
      pcout << std::string(80, '-') << std::endl;

      pcout << PRB_header << nu << std::endl;
      pcout << MPI_header << std::endl;
      pcout << SOL_header << std::endl;
      pcout << MET_header << std::endl;
      pcout << std::string(80, '=') << std::endl;
		std::vector<long> rusage_history(num_cycle);
		std::vector<double> wall_time(num_cycle);
		std::vector<double> cpu_time(num_cycle);
		const double final_time = 1;
		double time_step_size = 0.1;
		double current_time = 0.0;
		unsigned int num_time_step = floor(final_time/time_step_size);
		for (unsigned int cycle = 0; cycle < num_cycle; ++cycle)
		{
			pcout << std::string(80, '-') << std::endl;
			pcout << "Cycle " << cycle + 1 << std::endl;
			pcout << std::string(80, '-') << std::endl;

			deform_slab(time_step_size, current_time);
			Timer timer;
			pcout << "Set up system..." << std::endl;
			setup_system();
			pcout << "  Slab mesh: \t"
				<< triangulation.n_global_active_cells() << " cells" << std::endl;
			pcout << "  DoFHandler: \t"
				<< dof_handler.n_dofs() << " DoFs" << std::endl;
			assemble_system(false,true);
			rusage_history[cycle] = get_mem_usage()/1024.0;
			pcout << "  Mem usage: \t" << rusage_history[cycle] << " MB" << std::endl;
			pcout << "  Time step: \t"
				<< num_time_step << " steps" << std::endl;
			solve();
			new_initial_cond();
			calculate_errors();
			if (ifvtk)
				output_results(cycle,0);
			current_time += time_step_size;
			pcout << "Progress..." << std::endl;
			for (unsigned int time_step = 1; time_step <= num_time_step; ++time_step)
			{
				if (time_step == ceil(num_time_step/4))
					pcout << "  25%..." << std::endl;
				else if (time_step == ceil(num_time_step/2))
					pcout << "  50%..." << std::endl;
				else if (time_step == ceil(num_time_step*3/4))
					pcout << "  75%..." << std::endl;
				deform_slab(time_step_size, current_time);
				setup_system();
				assemble_system(false,false);
				solve();
				new_initial_cond();
				calculate_errors();
				if (time_step == num_time_step) {
					timer.stop();
					pcout << "  Done! (" << timer.wall_time() << "s)" << std::endl;
				}
				if (ifvtk)
					output_results(cycle,time_step);
				current_time += time_step_size;
			}
			timer.stop();
			wall_time[cycle] = timer.wall_time();
			cpu_time[cycle] = timer.cpu_time();
			timer.reset();
			for (unsigned int i = 0; i < error_list.size()-1; ++i) {
				error_list[error_list.size()-1] += error_list[i];
			}
			for (unsigned int i = 0; i < error_list.size(); ++i) {
				error_list[i] = std::sqrt(error_list[i]);
			}
			pcout << "Output results..." << std::endl;
			const double tnorm = error_list[7];
			pcout << "  Triple norm: " << tnorm << std::endl;
			convergence_table.add_value("cells", triangulation.n_global_active_cells());
			convergence_table.add_value("slabs", num_time_step);
			convergence_table.add_value("dofs", dof_handler.n_dofs());
			convergence_table.add_value("L2", error_list[0]);
			convergence_table.add_value("sH1", error_list[1]);
			convergence_table.add_value("tH1", error_list[2]);
			convergence_table.add_value("dif-jp", error_list[3]);
			convergence_table.add_value("adv-jp", error_list[4]);
			convergence_table.add_value("neum", error_list[5]);
			convergence_table.add_value("supg", error_list[6]);
			convergence_table.add_value("tnorm", error_list[7]);
			for (unsigned int i = 0; i < error_list.size(); ++i) {
				error_list[i] = 0.0;
			}
			if (cycle < num_cycle) {
				time_step_size = time_step_size/2;
				num_time_step = floor(final_time/time_step_size);
				current_time = 0.0;
			}
		}

		pcout << std::string(80, '=') << std::endl;
		pcout << "Convergence History: " << std::endl;
		convergence_table.set_scientific("L2", true);
		convergence_table.set_precision("L2", 1);
		convergence_table.set_scientific("sH1", true);
		convergence_table.set_precision("sH1", 1);
		convergence_table.set_scientific("tH1", true);
		convergence_table.set_precision("tH1", 1);
		convergence_table.set_scientific("dif-jp", true);
		convergence_table.set_precision("dif-jp", 1);
		convergence_table.set_scientific("adv-jp", true);
		convergence_table.set_precision("adv-jp", 1);
		convergence_table.set_scientific("neum", true);
		convergence_table.set_precision("neum", 1);
		convergence_table.set_scientific("supg", true);
		convergence_table.set_precision("supg", 1);
		convergence_table.set_scientific("tnorm", true);
		convergence_table.set_precision("tnorm", 1);

		convergence_table.evaluate_convergence_rates(
				"L2", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		convergence_table.evaluate_convergence_rates(
				"sH1", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		convergence_table.evaluate_convergence_rates(
				"tH1", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		convergence_table.evaluate_convergence_rates(
				"dif-jp", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		convergence_table.evaluate_convergence_rates(
				"adv-jp", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		convergence_table.evaluate_convergence_rates(
				"supg", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		convergence_table.evaluate_convergence_rates(
				"neum", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		convergence_table.evaluate_convergence_rates(
				"tnorm", "cells", ConvergenceTable::reduction_rate_log2, dim-1);
		pcout << std::string(80, '-') << std::endl;
		if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
			convergence_table.write_text(std::cout);
		pcout << std::string(80, '=') << std::endl;

			computing_timer.print_summary();
			computing_timer.reset();
		}

} // end of namespace SpaceTimeAdvecDiffuIPH



int main(int argc, char** argv)
{
	using namespace dealii;

	const unsigned int dim = 3;

	int opt;
	int opt_n = 0;
	int num_cycle = 0;
	int degree = 1;
	bool ifvtk = false;

	// Three command line options accepted:
	// n (nu), c (cycle), p (polynomial order) and o (generate vtks)
	while ( (opt = getopt(argc, argv, "n:c:p:o")) != -1 ) {
		switch ( opt ) {
			case 'n':
				opt_n = atoi(optarg);
				break;
			case 'c':
				num_cycle = atoi(optarg);
				break;
			case 'p':
				degree = atoi(optarg);
				break;
			case 'o':
				ifvtk = true;
				break;
			case '?':  // unknown option...
				std::cerr << "Unknown option: '" << char(optopt) << "'!" << std::endl;
				break;
		}
	}

	// Hacky solution to suppress PETSc unused option warnings:
	argc = 1;
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	const double nu = std::pow(10, -opt_n);

	try
	{
		SpaceTimeAdvecDiffuIPH::SpaceTimeHDG<dim> hdg_problem(degree, nu, num_cycle, ifvtk);
		hdg_problem.run ();
	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl
			<< std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
			<< exc.what() << std::endl
			<< "Aborting!" << std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl
			<< std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
			<< "Aborting!" << std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		return 1;
	}

	return 0;
}

