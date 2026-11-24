#pragma once

#include "steady_solver.hxx"

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace pyhygrotherm
{
    // Per-edge time series of boundary condition coefficients. A vector of
    // length 1 is a constant series, held once via the kernel's scalar BC
    // overloads; a longer vector is periodic and is cycled across the
    // timesteps (value at step t is series[t % series.size()]). Compact
    // series keep memory bounded on long runs: the kernel periodic BC for
    // an edge indexes every coefficient field by its own period -- memory
    // is proportional to the sum of the field periods, never num_steps and
    // never an lcm of the periods (which explodes on coprime periods). The
    // cyclical indexing lives in the periodic BC objects, driven by
    // IBCLinear2D::selectTimestep from
    // BoundaryConditions2D::forEachActiveBC.
    using TimestepConvectionInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>, std::vector<double>>;
    using TimestepTemperatureInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>>;
    using TimestepFluxInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>>;
    using TimestepRadiationInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>, std::vector<double>>;
    using TimestepLinearizedRadiationInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>, std::vector<double>>;

    // Hygrothermal boundary series: a convection edge that additionally
    // carries the air relative humidity driving the moisture-side Robin
    // exchange, a humidity-only Dirichlet pin, and a combined temperature and
    // humidity pin (TransientTypeBC UseTemperature+UseHumidity).
    using TimestepHumidConvectionInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>,
                 std::vector<double>, std::vector<double>>;
    using TimestepHumidityInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>>;
    using TimestepTemperatureHumidityInput =
      std::tuple<std::size_t, std::size_t, std::vector<double>,
                 std::vector<double>>;

    // (x, y) sample point of a material property table.
    using TablePoint = std::pair<double, double>;

    // Moisture-side properties of one solid material (nullopt marks a gas
    // cavity material, which has no hygrothermal extension). Field order:
    // density [kg/m3], porosity [0,1], heat capacity [J/(kg.K)],
    // vapor diffusion resistance factor mu [-], sorption curve (phi, w),
    // liquid transport curve (w, D_l), moisture-dependent conductivity table
    // keyed by water content (w [kg/m3], lambda [W/(m.K)]) with its
    // measurement temperature [C], and temperature-dependent conductivity
    // table (T [C], lambda [W/(m.K)]) with its measurement water content
    // [kg/m3].
    using HygrothermalMaterialInput =
      std::tuple<double, double, double, double,
                 std::vector<TablePoint>, std::vector<TablePoint>,
                 std::vector<TablePoint>, double,
                 std::vector<TablePoint>, double>;

    // Kernel PhysicsOptions as a flat flag tuple: exclude liquid transport,
    // exclude heat of evaporation, exclude capillary conduction, exclude
    // vapor diffusion conduction, exclude latent heat of fusion, and the
    // moisture+temperature dependent conductivity switch.
    using PhysicsOptionsInput =
      std::tuple<bool, bool, bool, bool, bool, bool>;

    // Per boundary-condition edge outward heat rate (W/m) keyed by 0-based
    // THMZ node positions.
    using TransientBoundaryEdgeRate = std::tuple<std::size_t, std::size_t, double>;

    // One completed timestep frame streamed to Python. Field rows are node
    // vectors in THMZ order. Dry solves leave the moisture columns zero.
    struct TransientFrame
    {
        std::size_t step = 0;              // size_t(-1) marks the t=0 frame
        double time = 0.0;                 // seconds from the initial state
        std::vector<double> temperature;
        std::vector<double> humidity;
        std::vector<double> water_content;
        // Ice content per node (kg/m3); needed to close the latent-heat
        // energy budget. Identically zero in dry solves.
        std::vector<double> ice_content;
        std::vector<std::array<double, 2>> heat_flux;
        std::vector<std::array<double, 2>> water_flux;
        std::vector<TransientBoundaryEdgeRate> edge_heat_rates;
        std::vector<TransientBoundaryEdgeRate> edge_water_rates;
        // Boundary exchange booked over the completed interval ending at this
        // frame: the per-edge sum of rate * dt across every accepted coupled
        // sub-step (including internal subdivision and shock-chain
        // micro-steps). J/m for heat, kg/m for water, outward-positive per
        // metre of depth; all zero on the t=0 frame. Populated only when
        // on_step is set (the kernel accumulators update unconditionally, so
        // the data exists even without a callback -- it is read into the
        // frame to keep the callback surface self-contained).
        std::vector<TransientBoundaryEdgeRate> edge_heat_exchange;
        std::vector<TransientBoundaryEdgeRate> edge_water_exchange;
        double temperature_error = 0.0;
        double humidity_error = 0.0;
    };

    struct TransientSolution
    {
        std::vector<double> temperature;
        std::vector<std::array<double, 2>> heat_flux;
        std::vector<double> humidity;
        std::vector<std::array<double, 2>> water_flux;
        std::size_t steps = 0;
        // Diagnostic: total cyclic-series slots the kernel periodic BCs
        // hold for this run. With compact series it equals the sum of the
        // per-field periods of every non-constant boundary series; a value
        // proportional to num_steps (or to an lcm of periods) would mean
        // the bounded-memory contract has regressed. Exposed so tests can
        // assert the bound cheaply.
        std::size_t bc_timestep_slots = 0;
    };

    // Drives a dry, thermal-only transient solve. The on_step callback is
    // invoked once for the t=0 initial frame (step == -1) and once after
    // each completed timestep, so results can be streamed out without
    // retaining the full series (the kernel's transientMultiStep accumulates
    // every step in memory, which does not scale for year-long runs).
    TransientSolution solve_transient(
      const std::vector<NodeInput> & nodes,
      const std::vector<QuadInput> & quads,
      const std::vector<MaterialInput> & materials,
      const std::vector<double> & densities,
      const std::vector<double> & heat_capacities,
      const std::vector<TimestepConvectionInput> & convection,
      const std::vector<TimestepTemperatureInput> & temperatures,
      const std::vector<TimestepFluxInput> & fluxes,
      const std::vector<TimestepRadiationInput> & radiations,
      const std::vector<TimestepLinearizedRadiationInput> & linearized_radiations,
      double initial_temperature,
      double initial_humidity,
      double time_step,
      std::size_t num_steps,
      double convergence_tolerance,
      double relaxation,
      const std::array<double, 3> & gravity = {0.0, -1.0, 0.0},
      const std::vector<CavitySurfaceEmissivityInput> &
        cavity_surface_emissivities = {},
      const std::function<void(const TransientFrame &)> & on_step = {});

    // Drives a coupled hygrothermal transient solve (MultiDomain with both
    // the thermal and the moisture domain active). Moisture boundary
    // semantics follow the TransientTypeBC rules: humid_convection edges get
    // the combined thermal+moisture Robin pair (the air humidity column
    // drives the Lewis-derived vapor transfer coefficient), humidities are
    // moisture Dirichlet pins, temperature_humidities pin both state
    // variables on the same edge, and dry convection/temperature/flux/
    // radiation edges behave exactly as in solve_transient.
    // hygro_materials parallels materials: nullopt where the element's
    // material is a gas cavity, otherwise the moisture-side properties.
    TransientSolution solve_hygrothermal(
      const std::vector<NodeInput> & nodes,
      const std::vector<QuadInput> & quads,
      const std::vector<MaterialInput> & materials,
      const std::vector<std::optional<HygrothermalMaterialInput>> &
        hygro_materials,
      const PhysicsOptionsInput & physics,
      const std::vector<TimestepConvectionInput> & convection,
      const std::vector<TimestepHumidConvectionInput> & humid_convection,
      const std::vector<TimestepTemperatureInput> & temperatures,
      const std::vector<TimestepHumidityInput> & humidities,
      const std::vector<TimestepTemperatureHumidityInput> &
        temperature_humidities,
      const std::vector<TimestepFluxInput> & fluxes,
      const std::vector<TimestepRadiationInput> & radiations,
      const std::vector<TimestepLinearizedRadiationInput> &
        linearized_radiations,
      double initial_temperature,
      double initial_humidity,
      double time_step,
      std::size_t num_steps,
      double convergence_tolerance,
      double relaxation,
      const std::array<double, 3> & gravity = {0.0, -1.0, 0.0},
      const std::vector<CavitySurfaceEmissivityInput> &
        cavity_surface_emissivities = {},
      const std::function<void(const TransientFrame &)> & on_step = {});
}   // namespace pyhygrotherm
