"""
.. autofunction:: gen_thermochem_code
.. autofunction:: get_thermochem_class

Internal Functionality
^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: equilibrium_constants_expr
.. autofunction:: rate_coefficient_expr
.. autofunction:: third_body_efficiencies_expr
.. autofunction:: troe_falloff_expr
.. autofunction:: falloff_function_expr
.. autofunction:: rate_of_progress_expr
.. autofunction:: production_rate_expr
"""

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
from functools import singledispatch

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
        return "self.usr_np.where(%s, %s, %s)" % (
            self.rec(expr.condition, PREC_NONE, *args, **kwargs),
            self.rec(expr.then, PREC_NONE, *args, **kwargs),
            self.rec(expr.else_, PREC_NONE, *args, **kwargs),
        )

    def map_call(self, expr, enclosing_prec, *args, **kwargs):
        return self.format(
            "self.usr_np.%s(%s)",
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

# {{{ Data-handling helper

def _zeros_like(argument):
    # FIXME: This mishandles NaNs.
    return 0 * argument

# }}}

# {{{ Equilibrium constants


def equilibrium_constants_expr(sol: ct.Solution, react: ct.Reaction, gibbs_rt):
    """Generate code for equilibrium constants.

    :returns: Equilibrium constant expression for reaction *react* in terms of
        the species Gibbs functions *gibbs_rt* as a
        :class:`pymbolic.primitives.Expression`
    """

    indices_reac = [sol.species_index(sp) for sp in react.reactants]
    indices_prod = [sol.species_index(sp) for sp in react.products]

    # Stoichiometric coefficients
    #nu_reac = [react.reactants[sp] for sp in react.reactants]
    #nu_prod = [react.products[sp] for sp in react.products]
    nu_reac = [sol.reactant_stoich_coeff(sol.species_index(sp), int(react.ID)-1)
               for sp in react.reactants]
    nu_prod = [sol.product_stoich_coeff(sol.species_index(sp), int(react.ID)-1)
               for sp in react.products]

    sum_r = sum(nu_reac_i * gibbs_rt[indices_reac_i]
            for indices_reac_i, nu_reac_i in zip(indices_reac, nu_reac))
    sum_p = sum(nu_prod_i * gibbs_rt[indices_prod_i]
            for indices_prod_i, nu_prod_i in zip(indices_prod, nu_prod))

    # Check if reaction is termolecular
    sum_nu_net = sum(nu_prod) - sum(nu_reac)
    if sum_nu_net < 0:
        # Three species on reactants side
        return sum_p - sum_nu_net*p.Variable("c0") - sum_r
    elif sum_nu_net > 0:
        # Three species on products side
        return sum_p - (sum_r - sum_nu_net*p.Variable("c0"))
    else:
        return sum_p - sum_r

# }}}


# {{{ Rate coefficients

def rate_coefficient_expr(rate_coeff: ct.Arrhenius, t):
    """
    :returns: The rate coefficient expression for *rate_coeff* in terms
        of the temperature *t* as a :class:`pymbolic.primitives.Expression`
    """
    # Rate parameters
    a = rate_coeff.pre_exponential_factor
    b = rate_coeff.temperature_exponent
    t_a = rate_coeff.activation_energy/ct.gas_constant
    if t_a == 0:
        # Weakly temperature-dependent rate
        return a * t**b
    else:
        # Modified Arrhenius
        return p.Variable("exp")(np.log(a)+b*p.Variable("log")(t)-t_a/t)


def third_body_efficiencies_expr(sol: ct.Solution, react: ct.Reaction, c):
    """
    :returns: The third-body concentration expression for reaction *react* in terms
        of the species concentrations *c* as a
        :class:`pymbolic.primitives.Expression`
    """
    efficiencies = [react.efficiencies[sp] for sp in react.efficiencies]
    indices_nondef = [sol.species_index(sp) for sp in react.efficiencies]
    indices_default = [i for i in range(sol.n_species) if i not in indices_nondef]
    sum_nondef = sum(eff_i * c[index_i] for eff_i, index_i
                     in zip(np.array(efficiencies), indices_nondef))
    sum_default = react.default_efficiency * sum(c[i] for i in indices_default)
    return sum_nondef + sum_default


def troe_falloff_expr(react: ct.Reaction, t):
    """
    :returns: The Troe falloff center expression for reaction *react* in terms of the
        temperature *t* as a :class:`pymbolic.primitives.Expression`
    """
    troe_params = react.falloff.parameters
    troe_1 = (1.0-troe_params[0])*p.Variable("exp")(-t/troe_params[1])
    troe_2 = troe_params[0]*p.Variable("exp")(-t/troe_params[2])
    if troe_params[3]:
        troe_3 = p.Variable("exp")(-troe_params[3]/t)
        return troe_1 + troe_2 + troe_3
    else:
        return troe_1 + troe_2


def falloff_function_expr(react: ct.Reaction, i, t, red_pressure, falloff_center):
    """
    :returns: Falloff function expression for reaction *react* in terms
        of the temperature *t*, reduced pressure *red_pressure*, and falloff center
        *falloff_center* as a :class:`pymbolic.primitives.Expression`
    """
    if react.falloff.falloff_type == "Troe":
        log_rp = p.Variable("log10")(red_pressure[i])
        c = -0.4-0.67*falloff_center[i]
        n = 0.75-1.27*falloff_center[i]
        f = (log_rp+c)/(n-0.14*(log_rp+c))
        return 10**((falloff_center[i])/(1+f**2))
    else:
        return 1

# }}}


# {{{ Rates of progress

def rate_of_progress_expr(sol: ct.Solution, react: ct.Reaction, c, k_fwd, k_eq):
    """
    :returns: Rate of progress expression for reaction *react* in terms of
        species concentrations *c* with rate coefficients *k_fwd* and equilbrium
        constants *k_eq* as a :class:`pymbolic.primitives.Expression`
    """
    indices_reac = [sol.species_index(sp) for sp in react.reactants]
    indices_prod = [sol.species_index(sp) for sp in react.products]

    if react.orders:
        nu_reac = [react.orders[sp] for sp in react.orders]
    else:
        nu_reac = [react.reactants[sp] for sp in react.reactants]

    r_fwd = np.prod([c[index]**nu for index, nu in zip(indices_reac, nu_reac)])

    if react.reversible:
        nu_prod = [react.products[sp] for sp in react.products]
        r_rev = np.prod([c[index]**nu for index, nu in zip(indices_prod, nu_prod)])
        # FIXME: It's not clear that this is available other than by this clunky,
        # string-parsing route
        reaction_index = int(react.ID)-1
        return k_fwd[reaction_index] * (r_fwd - k_eq[reaction_index] * r_rev)
    else:
        return k_fwd[int(react.ID)-1] * r_fwd

# }}}


# {{{ Species production rates

def production_rate_expr(sol: ct.Solution, species, r_net):
    """
    :returns: Species production rate for species *species* in terms of
        the net reaction rates of progress *r_net* as a
        :class:`pymbolic.primitives.Expression`
    """
    ones = _zeros_like(r_net[0]) + 1.0
    indices_fwd = [int(react.ID)-1 for react in sol.reactions()
                   if species in react.reactants]
    indices_rev = [int(react.ID)-1 for react in sol.reactions()
                   if species in react.products]
    nu_fwd = [sol.reactant_stoich_coeff(sol.species_index(species), react_index)
              for react_index in indices_fwd]
    nu_rev = [sol.product_stoich_coeff(sol.species_index(species), prod_index)
              for prod_index in indices_rev]
    sum_fwd = sum(nu*r_net[index] for nu, index in zip(nu_fwd, indices_fwd))
    sum_rev = sum(nu*r_net[index] for nu, index in zip(nu_rev, indices_rev))
    return (sum_rev - sum_fwd) * ones

# }}}


# {{{ main code template

code_tpl = Template(
    """\"""
.. autoclass:: Thermochemistry
\"""

import numpy as np


def _pyro_make_array(usr_np, res_list):
    \"""This works around (e.g.) numpy.exp not working with object arrays of numpy
    scalars. It defaults to making object arrays, however if an array consists of all
    scalars, it makes a "plain old" :class:`numpy.ndarray`.

    See ``this numpy bug <https://github.com/numpy/numpy/issues/18004>`__
    for more context.
    \"""

    from numbers import Number
    all_numbers = all(
        isinstance(e, Number)
        or (isinstance(e, usr_np.ndarray) and e.shape == ())
        for e in res_list)

    if all_numbers:
        return usr_np.array(res_list, dtype=usr_np.float64)

    result = usr_np.empty((len(res_list),), dtype=object)

    # 'result[:] = res_list' may look tempting, however:
    # https://github.com/numpy/numpy/issues/16564
    for idx in range(len(res_list)):
        result[idx] = res_list[idx]

    return result


def _pyro_norm(usr_np, argument, normord):
    \"""This works around numpy.linalg norm not working with scalars.

    If the argument is a regular ole number, it uses :func:`numpy.abs`,
    otherwise it uses ``usr_np.linalg.norm``.
    \"""
    # Wrap norm for scalars
    from numbers import Number
    if isinstance(argument, Number):
        return usr_np.abs(argument)
    if isinstance(argument, usr_np.ndarray) and argument.shape == ():
        return usr_np.abs(argument)
    return usr_np.linalg.norm(argument, normord)


def _pyro_zeros_like(argument):
    # FIXME: This is imperfect, as a NaN will stay a NaN.
    return 0 * argument


class Thermochemistry:
    \"""
    .. attribute:: model_name
    .. attribute:: num_elements
    .. attribute:: num_species
    .. attribute:: num_reactions
    .. attribute:: num_falloff
    .. attribute:: one_atm

        Returns 1 atm in SI units of pressure (Pa).

    .. attribute:: gas_constant
    .. attribute:: species_names
    .. attribute:: species_indices

    .. automethod:: get_specific_gas_constant
    .. automethod:: get_density
    .. automethod:: get_pressure
    .. automethod:: get_mix_molecular_weight
    .. automethod:: get_concentrations
    .. automethod:: get_mixture_specific_heat_cp_mass
    .. automethod:: get_mixture_specific_heat_cv_mass
    .. automethod:: get_mixture_enthalpy_mass
    .. automethod:: get_mixture_internal_energy_mass
    .. automethod:: get_species_specific_heats_r
    .. automethod:: get_species_enthalpies_rt
    .. automethod:: get_species_entropies_r
    .. automethod:: get_species_gibbs_rt
    .. automethod:: get_equilibrium_constants
    .. automethod:: get_temperature
    .. automethod:: __init__
    \"""

    def __init__(self, usr_np=np):
        \"""Initialize thermochemistry object for a mechanism.

        Parameters
        ----------
        usr_np
            :mod:`numpy`-like namespace providing at least the following functions,
            for any array ``X`` of the bulk array type:

            - ``usr_np.log(X)`` (like :data:`numpy.log`)
            - ``usr_np.log10(X)`` (like :data:`numpy.log10`)
            - ``usr_np.exp(X)`` (like :data:`numpy.exp`)
            - ``usr_np.where(X > 0, X_yes, X_no)`` (like :func:`numpy.where`)
            - ``usr_np.linalg.norm(X, np.inf)`` (like :func:`numpy.linalg.norm`)

            where the "bulk array type" is a type that offers arithmetic analogous
            to :class:`numpy.ndarray` and is used to hold all types of (potentialy
            volumetric) "bulk data", such as temperature, pressure, mass fractions,
            etc. This parameter defaults to *actual numpy*, so it can be ignored
            unless it is needed by the user (e.g. for purposes of
            GPU processing or automatic differentiation).

        \"""

        self.usr_np = usr_np
        self.model_name = ${repr(sol.source)}
        self.num_elements = ${sol.n_elements}
        self.num_species = ${sol.n_species}
        self.num_reactions = ${sol.n_reactions}
        self.num_falloff = ${
            sum(1 if isinstance(r, ct.FalloffReaction) else 0
            for r in sol.reactions())}

        self.one_atm = ${ct.one_atm}
        self.gas_constant = ${ct.gas_constant}
        self.big_number = 1.0e300

        self.species_names = ${sol.species_names}
        self.species_indices = ${
            dict([[sol.species_name(i), i]
                for i in range(sol.n_species)])}

        self.wts = ${str_np(sol.molecular_weights)}
        self.iwts = 1/self.wts

    def species_name(self, species_index):
        return self.species_name[species_index]

    def species_index(self, species_name):
        return self.species_indices[species_name]

    def get_specific_gas_constant(self, mass_fractions):
        return self.gas_constant * self.usr_np.dot(self.iwts, mass_fractions)

    def get_density(self, p, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return p * mmw / rt

    def get_pressure(self, rho, temperature, mass_fractions):
        mmw = self.get_mix_molecular_weight(mass_fractions)
        rt = self.gas_constant * temperature
        return rho * rt / mmw

    def get_mix_molecular_weight(self, mass_fractions):
        return 1/self.usr_np.dot(self.iwts, mass_fractions)

    def get_concentrations(self, rho, mass_fractions):
        return self.iwts * rho * mass_fractions

    def get_mass_average_property(self, mass_fractions, spec_property):
        return sum([mass_fractions[i] * spec_property[i] * self.iwts[i]
                    for i in range(self.num_species)])

    def get_mixture_specific_heat_cp_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature)
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_specific_heat_cv_mass(self, temperature, mass_fractions):
        cp0_r = self.get_species_specific_heats_r(temperature) - 1.0
        cpmix = self.get_mass_average_property(mass_fractions, cp0_r)
        return self.gas_constant * cpmix

    def get_mixture_enthalpy_mass(self, temperature, mass_fractions):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        hmix = self.get_mass_average_property(mass_fractions, h0_rt)
        return self.gas_constant * temperature * hmix

    def get_mixture_internal_energy_mass(self, temperature, mass_fractions):
        e0_rt = self.get_species_enthalpies_rt(temperature) - 1.0
        emix = self.get_mass_average_property(mass_fractions, e0_rt)
        return self.gas_constant * temperature * emix

    def get_species_specific_heats_r(self, temperature):
        return _pyro_make_array(self.usr_np, [
            % for sp in sol.species():
            ${cgm(poly_to_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_enthalpies_rt(self, temperature):
        return _pyro_make_array(self.usr_np, [
            % for sp in sol.species():
            ${cgm(poly_to_enthalpy_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_entropies_r(self, temperature):
        return _pyro_make_array(self.usr_np, [
            % for sp in sol.species():
                ${cgm(poly_to_entropy_expr(sp.thermo, "temperature"))},
            % endfor
                ])

    def get_species_gibbs_rt(self, temperature):
        h0_rt = self.get_species_enthalpies_rt(temperature)
        s0_r = self.get_species_entropies_r(temperature)
        return h0_rt - s0_r

    def get_equilibrium_constants(self, temperature):
        rt = self.gas_constant * temperature
        c0 = self.usr_np.log(self.one_atm / rt)

        g0_rt = self.get_species_gibbs_rt(temperature)
        return _pyro_make_array(self.usr_np, [
            %for react in sol.reactions():
                %if react.reversible:
                    ${cgm(equilibrium_constants_expr(
                        sol, react, Variable("g0_rt")))},
                %else:
                    -0.17364695002734*temperature,
                %endif
            %endfor
                ])

    def get_temperature(self, enthalpy_or_energy, t_guess, y, do_energy=False):
        if do_energy is False:
            pv_fun = self.get_mixture_specific_heat_cp_mass
            he_fun = self.get_mixture_enthalpy_mass
        else:
            pv_fun = self.get_mixture_specific_heat_cv_mass
            he_fun = self.get_mixture_internal_energy_mass

        num_iter = 500
        tol = 1.0e-6
        ones = _pyro_zeros_like(enthalpy_or_energy) + 1.0
        t_i = t_guess * ones

        for iter in range(num_iter):
            f = enthalpy_or_energy - he_fun(t_i, y)
            j = -pv_fun(t_i, y)
            dt = -f / j
            t_i += dt
            if _pyro_norm(self.usr_np, dt, np.inf) < tol:
                return t_i

        raise RuntimeError("Temperature iteration failed to converge")

    %if falloff_reactions:
    def get_falloff_rates(self, temperature, concentrations, k_fwd):
        ones = _pyro_zeros_like(temperature) + 1.0
        k_high = _pyro_make_array(self.usr_np, [
        %for react in falloff_reactions:
            ${cgm(rate_coefficient_expr(react.high_rate, Variable("temperature")))},
        %endfor
                ])

        k_low = _pyro_make_array(self.usr_np, [
        %for react in falloff_reactions:
            ${cgm(rate_coefficient_expr(react.low_rate, Variable("temperature")))},
        %endfor
                ])

        reduced_pressure = _pyro_make_array(self.usr_np, [
        %for i, react in enumerate(falloff_reactions):
            (${cgm(third_body_efficiencies_expr(
                sol, react, Variable("concentrations")))})*k_low[${i}]/k_high[${i}],
        %endfor
                            ])

        falloff_center = _pyro_make_array(self.usr_np, [
        %for react in falloff_reactions:
            %if react.falloff.falloff_type == "Troe":
            self.usr_np.log10(${cgm(troe_falloff_expr(
                                   react, Variable("temperature")))}),
            %else:
            1,
            %endif
        %endfor
                        ])

        falloff_function = _pyro_make_array(self.usr_np, [
        %for i, react in enumerate(falloff_reactions):
            ${cgm(falloff_function_expr(
                react, i, Variable("temperature"), Variable("reduced_pressure"),
                Variable("falloff_center")))},
        %endfor
                            ])*reduced_pressure/(1+reduced_pressure)

        %for i, react in enumerate(falloff_reactions):
        k_fwd[${int(react.ID)-1}] = k_high[${i}]*falloff_function[${i}]*ones
        %endfor
        return

    %endif
    def get_fwd_rate_coefficients(self, temperature, concentrations):
        ones = self.usr_np.zeros_like(temperature) + 1.0
        k_fwd = [
        %for react in sol.reactions():
        %if isinstance(react, ct.FalloffReaction):
            0*temperature,
        %else:
            ${cgm(rate_coefficient_expr(react.rate,
                                        Variable("temperature")))} * ones,
        %endif
        %endfor
                ]
        %if falloff_reactions:
        self.get_falloff_rates(temperature, concentrations, k_fwd)
        %endif

        %for react in three_body_reactions:
        k_fwd[${int(react.ID)-1}] *= (${cgm(third_body_efficiencies_expr(
            sol, react, Variable("concentrations")))})
        %endfor
        return _pyro_make_array(self.usr_np, k_fwd)

    def get_net_rates_of_progress(self, temperature, concentrations):
        k_fwd = self.get_fwd_rate_coefficients(temperature, concentrations)
        log_k_eq = self.get_equilibrium_constants(temperature)
        k_eq = self.usr_np.exp(log_k_eq)
        return _pyro_make_array(self.usr_np, [
                %for react in sol.reactions():
                    ${cgm(rate_of_progress_expr(sol, react,
                        Variable("concentrations"),
                        Variable("k_fwd"), Variable("k_eq")))},
                %endfor
               ])

    def get_net_production_rates(self, rho, temperature, mass_fractions):
        c = self.get_concentrations(rho, mass_fractions)
        r_net = self.get_net_rates_of_progress(temperature, c)
        ones = _pyro_zeros_like(r_net[0]) + 1.0
        return _pyro_make_array(self.usr_np, [
            %for sp in sol.species():
                ${cgm(production_rate_expr(sol, sp.name, Variable("r_net")))} * ones,
            %endfor
               ])""", strict_undefined=True)

# }}}


def gen_thermochem_code(sol: ct.Solution) -> str:
    """For the mechanism given by *sol*, return Python source code for a class conforming
    to a module containing a class called ``Thermochemistry`` adhering to the
    :class:`~pyrometheus.thermochem_example.Thermochemistry` interface.
    """
    return code_tpl.render(
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
        falloff_function_expr=falloff_function_expr,
        rate_of_progress_expr=rate_of_progress_expr,
        production_rate_expr=production_rate_expr,

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


def compile_class(code_str, class_name="Thermochemistry"):
    exec_dict = {}
    exec(compile(code_str, "<generated code>", "exec"), exec_dict)
    exec_dict["_MODULE_SOURCE_CODE"] = code_str

    return exec_dict[class_name]


def get_thermochem_class(sol: ct.Solution):
    """For the mechanism given by *sol*, return a class conforming to the
    :class:`~pyrometheus.thermochem_example.Thermochemistry` interface.
    """
    return compile_class(gen_thermochem_code(sol))


def cti_to_mech_file(cti_file_name, mech_file_name):
    """Write python file for mechanism specified by CTI file."""
    with open(mech_file_name, "w") as outf:
        code = gen_thermochem_code(ct.Solution(cti_file_name, "gas"))
        print(code, file=outf)

# vim: foldmethod=marker
