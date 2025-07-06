#include <pybind11/pybind11.h>
#include <cmath>
#include <random>

namespace py = pybind11;

enum OptionType { Call, Put };

class EuropeanOption {
public:
    double S, K, T, r, sigma;
    OptionType type;

    EuropeanOption(double S_, double K_, double T_, double r_, double sigma_, OptionType type_)
        : S(S_), K(K_), T(T_), r(r_), sigma(sigma_), type(type_) {}

    double payoff(double ST) const {
        if (type == Call) return std::max(ST - K, 0.0);
        else return std::max(K - ST, 0.0);
    }
};

class BlackScholesModel {
private:
    double norm_cdf(double x) const {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }
public:
    double price(const EuropeanOption& opt) const {
        double d1 = (std::log(opt.S / opt.K) + (opt.r + 0.5 * opt.sigma * opt.sigma) * opt.T) / (opt.sigma * std::sqrt(opt.T));
        double d2 = d1 - opt.sigma * std::sqrt(opt.T);
        if (opt.type == Call)
            return opt.S * norm_cdf(d1) - opt.K * std::exp(-opt.r * opt.T) * norm_cdf(d2);
        else
            return opt.K * std::exp(-opt.r * opt.T) * norm_cdf(-d2) - opt.S * norm_cdf(-d1);
    }
};

class MonteCarloModel {
private:
    int N;
public:
    MonteCarloModel(int num_simulations) : N(num_simulations) {}

    double price(const EuropeanOption& opt) const {
        double payoff_sum = 0.0;
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<> dist(0.0, 1.0);

        for (int i = 0; i < N; ++i) {
            double Z = dist(gen);
            double ST = opt.S * std::exp((opt.r - 0.5 * opt.sigma * opt.sigma) * opt.T + opt.sigma * std::sqrt(opt.T) * Z);
            double payoff = opt.payoff(ST);
            payoff_sum += payoff;
        }
        return std::exp(-opt.r * opt.T) * (payoff_sum / N);
    }
};

class BinomialModel {
private:
    int N;
public:
    BinomialModel(int steps) : N(steps) {}

    double price(const EuropeanOption& opt) const {
        double dt = opt.T / N;
        double u = std::exp(opt.sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp(opt.r * dt) - d) / (u - d);
        double discount = std::exp(-opt.r * dt);

        std::vector<double> option_values(N + 1);

        // Terminal payoffs
        for (int i = 0; i <= N; ++i) {
            double ST = opt.S * std::pow(u, N - i) * std::pow(d, i);
            option_values[i] = opt.payoff(ST);
        }

        // Backward induction
        for (int step = N - 1; step >= 0; --step) {
            for (int i = 0; i <= step; ++i) {
                option_values[i] = discount * (p * option_values[i] + (1 - p) * option_values[i + 1]);
            }
        }

        return option_values[0];
    }
};


PYBIND11_MODULE(option_pricing, m) {
    py::class_<EuropeanOption>(m, "EuropeanOption")
        .def(py::init<double, double, double, double, double, OptionType>())
        .def_readwrite("S", &EuropeanOption::S)
        .def_readwrite("K", &EuropeanOption::K)
        .def_readwrite("T", &EuropeanOption::T)
        .def_readwrite("r", &EuropeanOption::r)
        .def_readwrite("sigma", &EuropeanOption::sigma)
        .def_readwrite("type", &EuropeanOption::type);

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put)
        .export_values();

    py::class_<BlackScholesModel>(m, "BlackScholesModel")
        .def(py::init<>())
        .def("price", &BlackScholesModel::price);

    py::class_<MonteCarloModel>(m, "MonteCarloModel")
        .def(py::init<int>())
        .def("price", &MonteCarloModel::price);
    py::class_<BinomialModel>(m, "BinomialModel")
        .def(py::init<int>())
        .def("price", &BinomialModel::price);

}
