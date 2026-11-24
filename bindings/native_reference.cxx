#include <HygroThermFEM/HygroThermFEM.hxx>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main()
{
    using namespace HygroThermFEM;

    const std::array<std::array<double, 2>, 9> coordinates{{
      {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0},
      {0.0, 0.5}, {1.0, 0.5}, {2.0, 0.5},
      {0.0, 1.0}, {1.0, 1.0}, {2.0, 1.0}}};
    const std::array<std::array<std::size_t, 4>, 4> quads{{
      {0, 1, 4, 3}, {1, 2, 5, 4}, {3, 4, 7, 6}, {4, 5, 8, 7}}};
    struct Convection
    {
        std::size_t node1;
        std::size_t node2;
        double h;
        double airTemperature;
    };
    const std::array<Convection, 4> boundaries{{
      {0, 3, 10.0, 20.0}, {3, 6, 10.0, 20.0},
      {2, 5, 10.0, 0.0}, {5, 8, 10.0, 0.0}}};

    try
    {
        MultiDomain model({.performThermal = true, .performMoisture = false});
        model.setPhysicsOptions({.excludeHeatOfEvaporation = true,
                                 .excludeCapillaryConduction = true,
                                 .excludeVaporDiffusionConduction = true});
        constexpr double tolerance = 1e-10;
        model.setSolverSettings({.relaxationParameter = 1.0,
                                 .errorTolerance = tolerance,
                                 .maxNumberOfIterations = 32,
                                 .maxDivisions = 8,
                                 .numberOfSubtimesteps = 8});
        for(std::size_t index = 0; index < coordinates.size(); ++index)
        {
            model.nodes().createNode({.index = index + 1,
                                      .x = coordinates[index][0],
                                      .y = coordinates[index][1]});
        }
        auto & material = model.materials().createSolidMaterial("single material");
        material.setThermalConductivity(1.0);
        for(const auto & quad : quads)
        {
            model.createElement({.node1 = quad[0] + 1,
                                 .node2 = quad[1] + 1,
                                 .node3 = quad[2] + 1,
                                 .node4 = quad[3] + 1,
                                 .material = material.name()});
        }
        for(const auto & boundary : boundaries)
        {
            model.thermal().createBC_FixedHc(
              boundary.node1 + 1,
              boundary.node2 + 1,
              {boundary.airTemperature, boundary.h},
              false);
        }
        const auto solution = model.steadyState();
        constexpr double heatFlux = 20.0 / 2.2;
        if(!solution.converged || solution.temperatureError > tolerance
           || solution.temperature.size() != coordinates.size()
           || solution.heatFlux.size() != coordinates.size())
        {
            std::cerr << "native reference solve did not converge or return complete results\n";
            return 1;
        }
        for(std::size_t index = 0; index < coordinates.size(); ++index)
        {
            const auto expectedTemperature =
              20.0 - heatFlux * (0.1 + coordinates[index][0]);
            if(std::abs(solution.temperature[index] - expectedTemperature) > 1e-7
               || std::abs(solution.heatFlux[index].x - heatFlux) > 1e-6
               || std::abs(solution.heatFlux[index].y) > 1e-6)
            {
                std::cerr << "native reference mismatch at node " << index << '\n';
                return 1;
            }
        }

        std::cout << std::setprecision(17) << "{\"temperature\":[";
        for(std::size_t index = 0; index < solution.temperature.size(); ++index)
        {
            if(index != 0)
            {
                std::cout << ',';
            }
            std::cout << solution.temperature[index];
        }
        std::cout << "],\"heat_flux\":[";
        for(std::size_t index = 0; index < solution.heatFlux.size(); ++index)
        {
            if(index != 0)
            {
                std::cout << ',';
            }
            std::cout << '[' << solution.heatFlux[index].x << ',' << solution.heatFlux[index].y
                      << ']';
        }
        std::cout << ']';

        // Transient reference: same mesh and convection boundaries, a
        // uniformly warm initial state, and a unit-capacity material so the
        // characteristic time rho*cp*L^2/k is a few seconds and the series
        // converges to the steady solution within the run. The loop mirrors
        // MultiDomain::transientMultiStep but streams frames out, matching
        // the binding's bounded-memory contract.
        MultiDomain transientModel({.performThermal = true,
                                    .performMoisture = false});
        transientModel.setPhysicsOptions(
          {.excludeWaterLiquidTransportation = true,
           .excludeHeatOfEvaporation = true,
           .excludeCapillaryConduction = true,
           .excludeVaporDiffusionConduction = true,
           .excludeLatentHeatOfFusion = true});
        transientModel.setSolverSettings({.relaxationParameter = 1.0,
                                          .errorTolerance = tolerance,
                                          .maxNumberOfIterations = 32,
                                          .maxDivisions = 8,
                                          .numberOfSubtimesteps = 8});
        constexpr double initialTemperature = 10.0;
        for(std::size_t index = 0; index < coordinates.size(); ++index)
        {
            transientModel.nodes().createNode(
              {.index = index + 1,
               .x = coordinates[index][0],
               .y = coordinates[index][1],
               .state = {.temperature = initialTemperature,
                          .humidity = 0.0,
                          .pressure = 101325.0,
                          .liquidPercent = 1.0}});
        }
        auto & transientMaterial =
          transientModel.materials().createSolidMaterial("transient material");
        transientMaterial.setThermalConductivity(1.0);
        transientMaterial.setDensity(1.0);
        transientMaterial.setHeatCapacity(1.0);
        for(const auto & quad : quads)
        {
            transientModel.createElement({.node1 = quad[0] + 1,
                                          .node2 = quad[1] + 1,
                                          .node3 = quad[2] + 1,
                                          .node4 = quad[3] + 1,
                                          .material = transientMaterial.name()});
        }
        constexpr double timeStep = 0.5;
        constexpr std::size_t numberOfSteps = 400;
        for(const auto & boundary : boundaries)
        {
            std::vector<FixedBCHCCoefficients> coefficients;
            coefficients.reserve(numberOfSteps);
            for(std::size_t step = 0; step < numberOfSteps; ++step)
            {
                coefficients.emplace_back(boundary.airTemperature,
                                          boundary.h,
                                          0.0);
            }
            transientModel.thermal().createBC_FixedHc(
              boundary.node1 + 1, boundary.node2 + 1, coefficients, false);
        }
        std::vector<double> temperatures(coordinates.size(),
                                         initialTemperature);
        const std::vector<double> humidities(coordinates.size(), 0.0);
        std::vector<std::vector<double>> steps;
        std::vector<double> stepErrors;
        std::vector<double> currentHumidity = humidities;
        Solution lastSolution = transientModel.currentStateSolution();
        for(std::size_t step = 0; step < numberOfSteps; ++step)
        {
            const auto stepSolution = transientModel.transient(
              temperatures, currentHumidity, timeStep, step);
            if(stepSolution.temperature.size() != coordinates.size())
            {
                std::cerr << "transient reference returned incomplete results\n";
                return 1;
            }
            steps.push_back(stepSolution.temperature);
            stepErrors.push_back(stepSolution.temperatureError);
            temperatures = stepSolution.temperature;
            currentHumidity = stepSolution.humidity;
            lastSolution = stepSolution;
        }
        // After roughly fifty characteristic times the field must sit on the
        // steady solution within the same tolerances used above.
        for(std::size_t index = 0; index < coordinates.size(); ++index)
        {
            const auto expectedTemperature =
              20.0 - heatFlux * (0.1 + coordinates[index][0]);
            if(std::abs(lastSolution.temperature[index] - expectedTemperature)
                 > 1e-4)
            {
                std::cerr << "transient reference did not reach the steady "
                             "solution at node "
                          << index << '\n';
                return 1;
            }
        }
        std::cout << ",\"transient\":{\"time_step\":" << timeStep
                  << ",\"num_steps\":" << numberOfSteps
                  << ",\"initial_temperature\":" << initialTemperature
                  << ",\"steps\":[";
        for(std::size_t step = 0; step < steps.size(); ++step)
        {
            if(step != 0)
            {
                std::cout << ',';
            }
            std::cout << '[';
            for(std::size_t index = 0; index < steps[step].size(); ++index)
            {
                if(index != 0)
                {
                    std::cout << ',';
                }
                std::cout << steps[step][index];
            }
            std::cout << ']';
        }
        std::cout << "],\"temperature_errors\":[";
        for(std::size_t step = 0; step < stepErrors.size(); ++step)
        {
            if(step != 0)
            {
                std::cout << ',';
            }
            std::cout << stepErrors[step];
        }
        std::cout << "],\"final_heat_flux\":[";
        for(std::size_t index = 0; index < lastSolution.heatFlux.size(); ++index)
        {
            if(index != 0)
            {
                std::cout << ',';
            }
            std::cout << '[' << lastSolution.heatFlux[index].x << ','
                      << lastSolution.heatFlux[index].y << ']';
        }
        std::cout << "]}}\n";
    }
    catch(const std::exception & error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
