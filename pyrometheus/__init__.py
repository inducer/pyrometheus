__copyright__ = """
Copyright (C) 2020 Esteban Cisneros
Copyright (C) 2020 Andreas Kloeckner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from numbers import Number
from functools import singledispatch, reduce

import pymbolic.primitives as p
from pymbolic.mapper.stringifier import StringifyMapper, PREC_NONE, PREC_CALL
import cantera as ct
import numpy as np  # noqa: F401

from itertools import compress
from mako.template import Template


# {{{ code generation helpers


class CodeGenerationMapper(StringifyMapper):
    def map_constant(self, expr, enclosing_prec):
        return repr(expr)

    def map_if(self, expr, enclosing_prec, *args, **kwargs):
        return "np.where(%s, %s, %s)" % (
            self.rec(expr.condition, PREC_NONE, *args, **kwargs),
            self.rec(expr.then, PREC_NONE, *args, **kwargs),
            self.rec(expr.else_, PREC_NONE, *args, **kwargs),
        )

    def map_call(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(
            "np.%s(%s)",
            self.rec(expr.function, PREC_CALL, *args, **kwargs),
            self.join_rec(", ", expr.parameters, PREC_NONE, *args, **kwargs),
        )


def str_np_inner(ary):
    if isinstance(ary, Number):
        return repr(ary)
    elif ary.shape:
        return "[%s]" % (", ".join(str_np_inner(ary_i) for ary_i in ary))
    raise TypeError("invalid argument to str_np_inner")


def str_np(ary):
    return "np.array(%s)" % str_np_inner(ary)


# }}}


# {{{ polynomial processing


def nasa7_conditional(t, poly, part_gen):
    # FIXME: Should check minTemp, maxTemp
    return p.If(
        p.Comparison(t, ">", poly.coeffs[0]),
        part_gen(poly.coeffs[1:8], t),
        part_gen(poly.coeffs[8:15], t),
    )


@singledispatch
def poly_to_expr(poly):
    raise TypeError(f"unexpected argument type in poly_to_expr: {type(poly)}")


@poly_to_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    def gen(c, t):
        assert len(c) == 7
        return c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4

    return nasa7_conditional(p.Variable(arg_name), poly, gen)


@singledispatch
def poly_to_enthalpy_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_to_enthalpy_expr: "
            f"{type(poly)}")


@poly_to_enthalpy_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    def gen(c, t):
        assert len(c) == 7
        return (
            c[0]
            + c[1] / 2 * t
            + c[2] / 3 * t ** 2
            + c[3] / 4 * t ** 3
            + c[4] / 5 * t ** 4
            + c[5] / t
        )

    return nasa7_conditional(p.Variable(arg_name), poly, gen)


@singledispatch
def poly_to_entropy_expr(poly, arg_name):
    raise TypeError("unexpected argument type in poly_to_entropy_expr: "
            f"{type(poly)}")


@poly_to_entropy_expr.register
def _(poly: ct.NasaPoly2, arg_name):
    log = p.Variable("log")

    def gen(c, t):
        assert len(c) == 7
        return (
            c[0] * log(t)
            + c[1] * t
            + c[2] / 2 * t ** 2
            + c[3] / 3 * t ** 3
            + c[4] / 4 * t ** 4
            + c[6]
        )

    return nasa7_conditional(p.Variable(arg_name), poly, gen)


# }}}


# {{{ Equilibrium constants


def equilibrium_constants_expr(sol: ct.Solution, react: ct.Reaction, gibbs_rt):
    indices_reac = [sol.species_index(sp) for sp in react.reactants]
    indices_prod = [sol.species_index(sp) for sp in react.products]

    # Stoichiometric coefficients
    nu_reac = [react.reactants[sp] for sp in react.reactants]
    nu_prod = [react.products[sp] for sp in react.products]

    sum_r = sum(nu_reac_i * gibbs_rt[indices_reac_i]
            for indices_reac_i, nu_reac_i in zip(indices_reac, nu_reac))
    sum_p = sum(nu_prod_i * gibbs_rt[indices_prod_i]
            for indices_prod_i, nu_prod_i in zip(indices_prod, nu_prod))

    # Check if reaction is termolecular
    sum_nu_net = sum(nu_prod) - sum(nu_reac)
    if sum_nu_net < 0:
        # Three species on reactants side
        return sum_p + p.Variable("C0") - sum_r
    elif sum_nu_net > 0:
        # Three species on products side
        return sum_p - (sum_r + p.Variable("C0"))
    else:
        return sum_p - sum_r

# }}}


# {{{ Rate coefficients

def rate_coefficient_expr(rate_coeff: ct.Arrhenius, t):
    """This function returns rate-coefficient expressions"""
    # Rate parameters
    a = rate_coeff.pre_exponential_factor
    b = rate_coeff.temperature_exponent
    t_a = rate_coeff.activation_energy/ct.gas_constant
    if b == 0.0 and t_a == 0.0:
        # Constant rate
        return a
    elif b != 0.0 and t_a == 0.0:
        # Weakly temperature-dependent rate
        return a * t**b
    elif b == 0.0 and a != 0.0 and t_a != 0.0:
        # Classic Arrhenius rate
        return p.Variable("exp")(np.log(a)-t_a/t)
    else:
        # Modified Arrhenius
        return p.Variable("exp")(np.log(a)+b*p.Variable("log")(t)-t_a/t)


def third_body_efficiencies_expr(sol: ct.Solution, react: ct.Reaction, c):
    """This function returns an expression for the third-body rate factor"""
    efficiencies = [react.efficiencies[sp] for sp in react.efficiencies]
    indices_nondef = [sol.species_index(sp) for sp in react.efficiencies]
    indices_default = np.delete(np.linspace(0, sol.n_species-1, sol.n_species,
                                            dtype=int), indices_nondef)
    sum_nondef = sum(eff_i * c[index_i] for eff_i, index_i
                     in zip(efficiencies, indices_nondef))
    sum_default = react.default_efficiency * sum(c[i] for i in indices_default)
    return sum_nondef + sum_default


def troe_falloff_expr(react: ct.Reaction, t):
    """This function returns Troe-falloff expression"""
    troe_params = react.falloff.parameters
    troe_1 = (1.0-troe_params[0])*p.Variable("exp")(-t/troe_params[2])
    troe_2 = troe_params[0]*p.Variable("exp")(-t/troe_params[1])
    if troe_params[3] > 1.0e-16:
        troe_3 = p.Variable("exp")(-troe_params[3]/t)
        return troe_1 + troe_2 + troe_3
    else:
        return troe_1 + troe_2

# }}}


# {{{ Rates of progress

def rate_of_progress_expr(sol: ct.Solution, react: ct.Reaction, c, k_fwd, k_rev):
    """This function returns an expression for the reaction rate of progress"""
    indices_reac = [sol.species_index(sp) for sp in react.reactants]
    indices_prod = [sol.species_index(sp) for sp in react.products]

    if len(react.orders) > 0:
        nu_reac = [react.orders[sp] for sp in react.orders]
    else:
        nu_reac = [react.reactants[sp] for sp in react.reactants]

    r_fwd = reduce(lambda x, y: x*y, [c[index]**nu for index, nu
                                      in zip(indices_reac, nu_reac)])

    if react.reversible:
        nu_prod = [react.products[sp] for sp in react.products]
        r_rev = reduce(lambda x, y: x*y, [c[index]**nu for index, nu
                                          in zip(indices_prod, nu_prod)])
        return k_fwd[int(react.ID)-1] * r_fwd - k_rev[int(react.ID)-1] * r_rev
    else:
        return k_fwd[int(react.ID)-1] * r_fwd

# }}}


# {{{ Species production rates

def production_rate_expr(sol: ct.Solution, species, r_net):
    """This function returns an expression for a species production rate"""
    indices_fwd = [int(react.ID)-1 for react in sol.reactions()
                   if species in react.reactants]
    indices_rev = [int(react.ID)-1 for react in sol.reactions()
                   if species in react.products]
    sum_fwd = sum(r_net[index] for index in indices_fwd)
    sum_rev = sum(r_net[index] for index in indices_rev)
    return sum_rev - sum_fwd

# }}}


# {{{ main code template

code_tpl = Template(
    """
import numpy as np

class Thermochemistry:
    model_name    = ${repr(sol.source)}
    num_elements  = ${sol.n_elements}
    num_species   = ${sol.n_species}
    num_reactions = ${sol.n_reactions}
    num_falloff   = ${
        sum(1 if isinstance(r, ct.FalloffReaction) else 0
            for r in sol.reactions())}

    one_atm = 1.01325e5
    gas_constant = 8314.4621
    big_number = 1.0e300

    species_names = ${sol.species_names}
    species_indices = ${
        dict([[sol.species_name(i), i]
            for i in range(sol.n_species)])}

    wts = ${str_np(sol.molecular_weights)}
    iwts = 1/wts

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, Y):
        return self.gas_constant * np.dot( self.iwts, Y )

    def get_density(self, p, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return p * mmw / RT

    def get_pressure(self, rho, T, Y):
        mmw = self.get_mix_molecular_weight( Y )
        RT  = self.gas_constant * T
        return rho * RT / mmw

    def get_mix_molecular_weight(self, Y):
        return 1/np.dot( self.iwts, Y )

    def get_concentrations(self, rho, Y):
        return self.iwts * rho * Y

    def get_mixture_specific_heat_cp_mass(self, T, Y):
        return self.gas_constant * np.sum(
            self.get_species_specific_heats_R(T)* Y * self.iwts)

    def get_mixture_specific_heat_cv_mass(self, T, Y):
        cp0_R = self.get_species_specific_heats_R( T ) - 1.0
        return self.gas_constant * np.sum(Y * cp0_R * self.iwts)

    def get_mixture_enthalpy_mass(self, T, Y):
        h0_RT = self.get_species_enthalpies_RT( T )
        return self.gas_constant * T * np.sum(Y * h0_RT * self.iwts)

    def get_mixture_internal_energy_mass(self, T, Y):
        e0_RT = self.get_species_enthalpies_RT( T ) - 1.0
        return self.gas_constant * T * np.sum(Y * e0_RT * self.iwts)

    def get_species_specific_heats_R(self, T):
        return np.array([
            % for sp in sol.species():
            ${cgm(poly_to_expr(sp.thermo, "T"))},
            % endfor
            ])

    def get_species_enthalpies_RT(self, T):
        return np.array([
            % for sp in sol.species():
            ${cgm(poly_to_enthalpy_expr(sp.thermo, "T"))},
            % endfor
            ])

    def get_species_entropies_R(self, T):
        return np.array([
            % for sp in sol.species():
                ${cgm(poly_to_entropy_expr(sp.thermo, "T"))},
            % endfor
            ])

    def get_species_gibbs_RT(self, T):
        h0_RT = self.get_species_enthalpies_RT(T)
        s0_R  = self.get_species_entropies_R(T)
        return h0_RT - s0_R

    def get_equilibrium_constants(self, T):
        RT = self.gas_constant * T
        C0 = np.log( self.one_atm / RT )

        g0_RT = self.get_species_gibbs_RT( T )
        return np.array([
            %for react in sol.reactions():
                %if react.reversible:
                    ${cgm(equilibrium_constants_expr(
                        sol, react, Variable("g0_RT")))},
                %else:
                    0*T,
                %endif
            %endfor
            ])

    def get_temperature(self, H_or_E, T_guess, Y, do_energy=False):
        if do_energy == False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        T_i = T_guess
        dT = 1.0
        F  = H_or_E
        J  = 0.0

        for iter in range( 0, num_iter ):
            F    -= he_fun( T_i, Y )
            J    -= pv_fun( T_i, Y )
            dT    = - F / J
            T_i  += dT
            if np.abs( dT ) < tol:
                T = T_i
                break
            F = H_or_E
            J = 0.0

        T = T_i

        return T

    def get_falloff_rates(self, T, C, k_fwd):
        k_high = np.zeros(self.num_falloff)
        k_low = np.zeros(self.num_falloff)
        reduced_pressure = np.zeros(self.num_falloff)
        falloff_function = np.zeros(self.num_falloff)

         %for i, react in zip(range(len(falloff_reactions)), falloff_reactions):
        k_high[${i}] = ${cgm(rate_coefficient_expr(react.high_rate, Variable("T")))}
        k_low[${i}] = ${cgm(rate_coefficient_expr(react.low_rate, Variable("T")))}
        reduced_pressure[${i}] = (${cgm(third_body_efficiencies_expr(
            sol, react, Variable("C")))})*k_low[${i}]/k_high[${i}]
        %if react.falloff.falloff_type == "Troe":
        falloff_center = np.log10(
            ${cgm(troe_falloff_expr(react, Variable("T")))})
        log_rp = np.log10(reduced_pressure[${i}])
        c = -0.4-0.67*falloff_center
        n = 0.75-1.27*falloff_center
        f = (log_rp+c)/(n-0.14*(log_rp+c))
        falloff_function[${i}] = 10.0**(falloff_center/(1.0+f**2))
        %else:
        falloff_function[${i}] = 1.0
        %endif

        %endfor

        falloff_function *= reduced_pressure/(reduced_pressure+1.0)

        %for i, react in zip(range(len(falloff_reactions)), falloff_reactions):
        k_fwd[${int(react.ID)-1}] = k_high[${i}]*falloff_function[${i}]
        %endfor
        return

    def get_fwd_rate_coefficients(self, T, C):
        k_fwd = np.zeros(self.num_reactions)
        %if len(falloff_reactions) > 0:
        self.get_falloff_rates(T, C, k_fwd)
        %endif
        %for react in non_falloff_reactions:
        k_fwd[${int(react.ID)-1}] = ${cgm(rate_coefficient_expr(
            react.rate, Variable("T")))}
        %endfor

        %for react in three_body_reactions:
        k_fwd[${int(react.ID)-1}] *= (${cgm(third_body_efficiencies_expr(
            sol, react, Variable("C")))})
        %endfor

        return k_fwd

    def get_net_rates_of_progress(self, T, C):
        k_fwd = self.get_rate_coefficients(T, C)
        k_rev = k_fwd * np.exp(self.get_equilibrium_constants(T))
        return np.array([
            %for react in sol.reactions():
                ${cgm(rate_of_progress_expr(sol, react, Variable("C"),
                    Variable("k_fwd"), Variable("k_rev")))},
            %endfor
            ])

    def get_net_production_rates(self, rho, T, Y):
        C = self.get_concentrations(rho, Y)
        r_net = self.get_net_rates_of_progress(T, C)
        return np.array([
            %for sp in sol.species():
                ${cgm(production_rate_expr(sol, sp.name, Variable("r_net")))},
            %endfor
            ])

""", strict_undefined=True)

# }}}


def gen_python_code(sol: ct.Solution):
    code = code_tpl.render(
        ct=ct,
        sol=sol,

        str_np=str_np,
        cgm=CodeGenerationMapper(),
        Variable=p.Variable,

        poly_to_expr=poly_to_expr,
        poly_to_enthalpy_expr=poly_to_enthalpy_expr,
        poly_to_entropy_expr=poly_to_entropy_expr,
        equilibrium_constants_expr=equilibrium_constants_expr,
        rate_coefficient_expr=rate_coefficient_expr,
        third_body_efficiencies_expr=third_body_efficiencies_expr,
        troe_falloff_expr=troe_falloff_expr,
        rate_of_progress_expr=rate_of_progress_expr,
        production_rate_expr=production_rate_expr,

        # falloff_flags=[isinstance(r, ct.FalloffReaction)
        #                for r in sol.reactions()],
        # non_falloff_flags=[not isinstance(r, ct.FalloffReaction)
        #                    for r in sol.reactions()],
        # third_body_flags=[isinstance(r, ct.ThreeBodyReaction)
        #                   for r in sol.reactions()],

        falloff_reactions=list(compress(sol.reactions(),
                                        [isinstance(r, ct.FalloffReaction)
                                         for r in sol.reactions()])),
        non_falloff_reactions=list(compress(sol.reactions(),
                                            [not isinstance(r, ct.FalloffReaction)
                                             for r in sol.reactions()])),
        three_body_reactions=list(compress(sol.reactions(),
                                           [isinstance(r, ct.ThreeBodyReaction)
                                            for r in sol.reactions()])),
    )
    print(code)
    exec_dict = {}
    exec(compile(code, "<generated code>", "exec"), exec_dict)
    return exec_dict["Thermochemistry"]


# vim: foldmethod=marker
