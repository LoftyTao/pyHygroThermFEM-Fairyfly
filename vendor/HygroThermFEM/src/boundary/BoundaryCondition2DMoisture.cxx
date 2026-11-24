#include "BoundaryCondition2DMoisture.hxx"
#include "VectorOperators.hxx"

namespace HygroThermFEM
{
    /////////////////////////////////////////////////////
    /// MoistureBCTARPHc
    /////////////////////////////////////////////////////
    MoistureBCTARPHc::MoistureBCTARPHc(Nodes & nodePool,
                                       Materials & materialPool,
                                       const size_t index1,
                                       const size_t index2,
                                       const std::string & materialName,
                                       const TARPCoefficients & varHCCoeff,
                                       const double surfaceTilt) :
        IMoistureBC(nodePool,
                    materialPool,
                    index1,
                    index2,
                    materialName,
                    varHCCoeff.AirHumidity,
                    varHCCoeff.AirTemperature,
                    ConvectionModelFactory::createTARPFilmCoefficient(
                      m_Nodes, varHCCoeff.AirTemperature, surfaceTilt))
    {}

    /////////////////////////////////////////////////////
    /// MoistureBCASHRAEInside
    /////////////////////////////////////////////////////
    MoistureBCASHRAEInside::MoistureBCASHRAEInside(Nodes & nodePool,
                                                   Materials & materialPool,
                                                   const size_t index1,
                                                   const size_t index2,
                                                   const std::string & materialName,
                                                   const ASHRAEInsideCoefficients & coeffs,
                                                   const double surfaceHeight,
                                                   const double surfaceTilt) :
        IMoistureBC(
          nodePool,
          materialPool,
          index1,
          index2,
          materialName,
          coeffs.AirHumidity,
          coeffs.AirTemperature,
          ConvectionModelFactory::createASHRAEInsideFilmCoefficient(
            m_Nodes, coeffs.AirTemperature, surfaceTilt, surfaceHeight, coeffs.AirPressure))
    {}

    /////////////////////////////////////////////////////
    /// MoistureBCASHRAEOutside
    /////////////////////////////////////////////////////
    MoistureBCASHRAEOutside::MoistureBCASHRAEOutside(Nodes & nodePool,
                                                     Materials & materialPool,
                                                     const size_t index1,
                                                     const size_t index2,
                                                     const std::string & materialName,
                                                     const ASHRAEOutsideCoefficients & coeffs) :
        IMoistureBC(
          nodePool,
          materialPool,
          index1,
          index2,
          materialName,
          coeffs.AirHumidity,
          coeffs.AirTemperature,
          ConvectionModelFactory::createASHRAEOutsideFilmCoefficient(m_Nodes, coeffs.WindSpeed))
    {}

    /////////////////////////////////////////////////////
    /// MoistureYazdanianKlemsBC
    /////////////////////////////////////////////////////
    MoistureYazdanianKlemsBC::MoistureYazdanianKlemsBC(Nodes & nodePool,
                                                       Materials & materialPool,
                                                       const size_t index1,
                                                       const size_t index2,
                                                       const std::string & materialName,
                                                       const YazdanianKlemsCoefficients & coeffs) :
        IMoistureBC(nodePool,
                    materialPool,
                    index1,
                    index2,
                    materialName,
                    coeffs.AirHumidity,
                    coeffs.AirTemperature,
                    ConvectionModelFactory::createYazdanianKlemsFilmCoefficient(
                      m_Nodes, coeffs.AirTemperature, coeffs.WindSpeed, coeffs.WindDir))
    {}

    /////////////////////////////////////////////////////
    /// MoistureKimuraBC
    /////////////////////////////////////////////////////
    MoistureKimuraBC::MoistureKimuraBC(Nodes & nodePool,
                                       Materials & materialPool,
                                       const size_t index1,
                                       const size_t index2,
                                       const std::string & materialName,
                                       const KimuraCoefficients & coeffs) :
        IMoistureBC(nodePool,
                    materialPool,
                    index1,
                    index2,
                    materialName,
                    coeffs.AirHumidity,
                    coeffs.AirTemperature,
                    ConvectionModelFactory::createKimuraFilmCoefficient(
                      m_Nodes, coeffs.WindSpeed, coeffs.WindDir))
    {}

    /////////////////////////////////////////////////////
    /// MoistureBCFixedHc
    /////////////////////////////////////////////////////
    MoistureBCFixedHc::MoistureBCFixedHc(Nodes & nodePool,
                                         Materials & materialPool,
                                         const size_t index1,
                                         const size_t index2,
                                         const std::string & materialName,
                                         const FixedBCHCCoefficients & fixedBchcCoefficients) :
        IMoistureBC(nodePool,
                    materialPool,
                    index1,
                    index2,
                    materialName,
                    fixedBchcCoefficients.AirHumidity,
                    fixedBchcCoefficients.AirTemperature,
                    ConvectionModelFactory::createFixedFilmCoefficient(
                      m_Nodes, fixedBchcCoefficients.ConvectionCoefficient))
    {}

    /////////////////////////////////////////////////////
    /// MoisturePeriodicConvectionBC
    /////////////////////////////////////////////////////
    MoisturePeriodicConvectionBC::MoisturePeriodicConvectionBC(
      Nodes & nodePool,
      Materials & materialPool,
      const size_t index1,
      const size_t index2,
      const std::string & materialName,
      const FixedHcPeriodicCoefficients & coefficients) :
        IMoistureBC(nodePool,
                    materialPool,
                    index1,
                    index2,
                    materialName,
                    coefficients.AirHumidities.at(0),
                    coefficients.AirTemperatures.at(0),
                    std::make_unique<PeriodicFilmCoefficient>(
                      m_Nodes, coefficients.ConvectionCoefficients)),
        m_AirTemperatures(coefficients.AirTemperatures),
        m_AirHumidities(coefficients.AirHumidities)
    {}

    void MoisturePeriodicConvectionBC::selectTimestep(const size_t timestepIndex)
    {
        m_Timestep = timestepIndex;
        m_ConvectiveCoeffCalc->selectTimestep(timestepIndex);
    }

    double MoisturePeriodicConvectionBC::airTemperature() const
    {
        return m_AirTemperatures.at(m_Timestep);
    }

    double MoisturePeriodicConvectionBC::airHumidity() const
    {
        return m_AirHumidities.at(m_Timestep);
    }

    /////////////////////////////////////////////////////
    /// MoisturePeriodicFixedHumidity
    /////////////////////////////////////////////////////
    MoisturePeriodicFixedHumidity::MoisturePeriodicFixedHumidity(
      Nodes & nodePool,
      Materials & materialPool,
      const size_t index1,
      const size_t index2,
      const std::string & materialName,
      PeriodicSeries humidities) :
        MoisturePeriodicConvectionBC(
          nodePool, materialPool, index1, index2, materialName,
          {{{0.0}}, {{hugeFilmCoefficient}}, std::move(humidities)})
    {}

    BCVector MoisturePeriodicFixedHumidity::R_Vector() const
    {
        // Node-temperature saturation concentration on BOTH sides of the penalty
        // row (the matrix side already uses it), so the pinned quantity is the
        // humidity itself -- see MoistureBCFixedHumidity.
        BCVector concentration{};
        for(std::size_t node = 0; node < numOfBCNodes; ++node)
        {
            concentration[node] = saturationConcentrationAtTemperature(
              m_Nodes[node].property(Variable::temperature));
        }
        const auto gconv = concentration
                           * m_ConvectiveCoeffCalc->waterVaporTransferCoefficient()
                           * airHumidity();
        return m_PsiVector * gconv;
    }

    /////////////////////////////////////////////////////
    /// MoistureBCFixedHumidity
    /////////////////////////////////////////////////////
    BCVector MoistureBCFixedHumidity::R_Vector() const
    {
        // Node-temperature saturation concentration on BOTH sides of the penalty
        // row (the matrix side already uses it), so the pinned quantity is the
        // humidity itself -- see the class comment.
        BCVector concentration{};
        for(std::size_t node = 0; node < numOfBCNodes; ++node)
        {
            concentration[node] = saturationConcentrationAtTemperature(
              m_Nodes[node].property(Variable::temperature));
        }
        const auto gconv = concentration
                           * m_ConvectiveCoeffCalc->waterVaporTransferCoefficient()
                           * m_AirHumidity;
        return m_PsiVector * gconv;
    }
}   // namespace HygroThermFEM
