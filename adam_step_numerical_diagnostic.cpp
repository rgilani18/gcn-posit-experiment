#include <cmath>
#include <iomanip>
#include <iostream>

#include <sw/universal/number/cfloat/cfloat.hpp>
#include <sw/universal/number/posit/posit.hpp>

using FP16 = sw::universal::cfloat<16, 5>;
using Posit16 = sw::universal::posit<16, 1>;

int main() {
    // Synthetic test values: this is a validation test, not a GCN result.
    const long double p0 = 0.25L;
    const long double grad = 0.0123L;
    const long double m0 = 0.001L;
    const long double v0 = 0.00001L;

    const long double beta1 = 0.9L;
    const long double beta2 = 0.999L;
    const long double lr = 0.01L;
    const long double eps = 1e-4L;
    const long double wd = 5e-4L;
    const int epoch = 1;

    // High-precision reference for this synthetic step.
    const long double g_ref = grad + wd * p0;
    const long double m_ref = beta1 * m0 + (1.0L - beta1) * g_ref;
    const long double v_ref = beta2 * v0 + (1.0L - beta2) * g_ref * g_ref;
    const long double bc1_ref = 1.0L - std::pow(beta1, epoch);
    const long double bc2_ref = 1.0L - std::pow(beta2, epoch);
    const long double mh_ref = m_ref / bc1_ref;
    const long double vh_ref = v_ref / bc2_ref;
    const long double step_ref = lr * mh_ref / (std::sqrt(vh_ref) + eps);
    const long double p_ref = p0 - step_ref;

    // Original FP16: FP16 parameters, gradients, moments, constants and update.
    const FP16 p16 = FP16(p0);
    const FP16 grad16 = FP16(grad);
    const FP16 m016 = FP16(m0);
    const FP16 v016 = FP16(v0);
    const FP16 b1_16 = FP16(0.9);
    const FP16 b2_16 = FP16(0.999);
    const FP16 lr16 = FP16(0.01);
    const FP16 eps16 = FP16(1e-4);
    const FP16 wd16 = FP16(5e-4);

    const FP16 g16 = grad16 + wd16 * p16;
    const FP16 m16 = b1_16 * m016 + (FP16(1) - b1_16) * g16;
    const FP16 v16 = b2_16 * v016 + (FP16(1) - b2_16) * g16 * g16;
    const FP16 mh16 = m16 / FP16(1.0 - std::pow(0.9, epoch));
    const FP16 vh16 = v16 / FP16(1.0 - std::pow(0.999, epoch));
    const FP16 denom16 = FP16(std::sqrt(static_cast<double>(vh16))) + eps16;
    const FP16 p16_new = p16 - lr16 * mh16 / denom16;

    std::cout << std::setprecision(18) << std::scientific;
    std::cout << "FP16 intermediates:"
              << " p=" << static_cast<double>(p16)
              << " grad=" << static_cast<double>(grad16)
              << " g=" << static_cast<double>(g16)
              << " m=" << static_cast<double>(m16)
              << " v=" << static_cast<double>(v16)
              << " m_hat=" << static_cast<double>(mh16)
              << " v_hat=" << static_cast<double>(vh16)
              << " denominator=" << static_cast<double>(denom16)
              << " lr=" << static_cast<double>(lr16)
              << " updated_p=" << static_cast<double>(p16_new)
              << '\\n';

    // FP16 parameters with double-precision Adam arithmetic and state.
    const double p_d = static_cast<double>(p16);
    const double grad_d = static_cast<double>(grad16);
    const double g_d = grad_d + 5e-4 * p_d;
    const double m_d = 0.9 * static_cast<double>(m016) + (1.0 - 0.9) * g_d;
    const double v_d = 0.999 * static_cast<double>(v016) +
                       (1.0 - 0.999) * g_d * g_d;
    const double mh_d = m_d / (1.0 - std::pow(0.9, epoch));
    const double vh_d = v_d / (1.0 - std::pow(0.999, epoch));
    const double step_d = 0.01 * mh_d / (std::sqrt(vh_d) + 1e-4);
    const FP16 p16_double_new = FP16(p_d - step_d);

    // Posit16: Posit16 parameters, gradients, moments, constants and update.
    const Posit16 pp = Posit16(p0);
    const Posit16 gp = Posit16(grad);
    const Posit16 mp0 = Posit16(m0);
    const Posit16 vp0 = Posit16(v0);
    const Posit16 b1p = Posit16(0.9);
    const Posit16 b2p = Posit16(0.999);
    const Posit16 lrp = Posit16(0.01);
    const Posit16 epsp = Posit16(1e-4);
    const Posit16 wdp = Posit16(5e-4);

    const Posit16 g_p = gp + wdp * pp;
    const Posit16 m_p = b1p * mp0 + (Posit16(1) - b1p) * g_p;
    const Posit16 v_p = b2p * vp0 + (Posit16(1) - b2p) * g_p * g_p;
    const Posit16 mh_p = m_p / Posit16(1.0 - std::pow(0.9, epoch));
    const Posit16 vh_p = v_p / Posit16(1.0 - std::pow(0.999, epoch));
    const Posit16 denom_p = Posit16(std::sqrt(static_cast<double>(vh_p))) + epsp;
    const Posit16 pp_new = pp - lrp * mh_p / denom_p;

    const long double p16_result = static_cast<long double>(static_cast<double>(p16_new));
    const long double p16d_result = static_cast<long double>(static_cast<double>(p16_double_new));
    const long double posit_result = static_cast<long double>(static_cast<double>(pp_new));

    auto report = [&](const char* name, long double value) {
        const long double abs_err = std::fabs(value - p_ref);
        const long double rel_err = abs_err / std::max(std::fabs(p_ref), 1e-30L);
        std::cout << name
                  << " updated_parameter=" << value
                  << " abs_error_vs_long_double=" << abs_err
                  << " rel_error_vs_long_double=" << rel_err << '\n';
    };

    std::cout << std::setprecision(18) << std::scientific;
    std::cout << "Synthetic one-step Adam diagnostic (not a GCN result)\n";
    std::cout << "long_double_reference_parameter=" << p_ref << '\n';
    report("FP16", p16_result);
    report("FP16_DoubleAdam", p16d_result);
    report("Posit16", posit_result);
    return 0;
}
