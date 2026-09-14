# Directed rounding for scalars, without touching the hardware rounding mode.
#
# Every certified scalar in the package is computed with these helpers. Each one performs the
# operation in round-to-nearest and then steps one unit in the last place outward
# (`nextfloat` for an upper bound, `prevfloat` for a lower bound). For a correctly rounded
# operation (IEEE +, -, *, /, sqrt, and BigFloat arithmetic) the round-to-nearest result is within
# half a unit in the last place of the exact value, so the step makes it a bound; this holds in
# the subnormal range too, and an overflow to ±Inf gives the trivial bound. The cost is one ulp
# per operation compared with true directed rounding, and the helpers are safe inside
# `Threads.@threads`, which a change of the hardware rounding mode is not.
#
# Ball matrix products are not done here: they are BallArithmetic's (outward rounded).

"Unit roundoff `u = eps(T)/2`."
unit_roundoff(::Type{T}) where {T <: AbstractFloat} = eps(T) / 2
unit_roundoff(::Type{Complex{T}}) where {T} = unit_roundoff(T)

"Smallest positive subnormal (the absolute error of an underflowing product is at most half of it)."
eta_min(::Type{Float64}) = 2.0^-1074
eta_min(::Type{Float32}) = Float32(2.0^-149)
eta_min(::Type{Complex{T}}) where {T} = eta_min(T)

@inline addu(a::T, b::T) where {T <: AbstractFloat} = nextfloat(a + b)
@inline addd(a::T, b::T) where {T <: AbstractFloat} = prevfloat(a + b)
@inline subu(a::T, b::T) where {T <: AbstractFloat} = nextfloat(a - b)
@inline subd(a::T, b::T) where {T <: AbstractFloat} = prevfloat(a - b)
@inline mulu(a::T, b::T) where {T <: AbstractFloat} = nextfloat(a * b)
@inline muld(a::T, b::T) where {T <: AbstractFloat} = prevfloat(a * b)
@inline divu(a::T, b::T) where {T <: AbstractFloat} = nextfloat(a / b)
@inline divd(a::T, b::T) where {T <: AbstractFloat} = prevfloat(a / b)
@inline sqrtu(a::T) where {T <: AbstractFloat} = a <= 0 ? (a == 0 ? zero(T) : T(NaN)) :
                                                  nextfloat(sqrt(a))
@inline sqrtd(a::T) where {T <: AbstractFloat} = a <= 0 ? zero(T) : max(zero(T), prevfloat(sqrt(a)))

# mixed-type conveniences (the other argument is converted exactly: integers below 2^53)
for f in (:addu, :addd, :subu, :subd, :mulu, :muld, :divu, :divd)
    @eval @inline $f(a::Real, b::Real) = $f(promote(float(a), float(b))...)
end

"Upper bound of `a + b + c + …`."
sumu(xs) = foldl(addu, xs; init = 0.0)

"Upper bound of `|x|` (exact for real `x`)."
@inline absu(x::Real) = abs(float(x))
@inline absu(z::Complex) = sqrtu(addu(mulu(real(z), real(z)), mulu(imag(z), imag(z))))
"Lower bound of `|x|`."
@inline absd(x::Real) = abs(float(x))
@inline absd(z::Complex) = sqrtd(addd(max(0.0, muld(real(z), real(z))),
    max(0.0, muld(imag(z), imag(z)))))

"Upper bound of `x^e`, `x ≥ 0`, `e ≥ 0` an integer."
function powu(x::T, e::Integer) where {T <: AbstractFloat}
    r = one(T)
    for _ in 1:e
        r = mulu(r, x)
    end
    return r
end
"Lower bound of `x^e`, `x ≥ 0`."
function powd(x::T, e::Integer) where {T <: AbstractFloat}
    r = one(T)
    for _ in 1:e
        r = max(zero(T), muld(r, x))
    end
    return r
end

"Upper bound of `x^(1/e)`, `x ≥ 0`: step up from the floating-point root until `r^e ≥ x` holds."
function rootu(x::Float64, e::Integer)
    x <= 0 && return 0.0
    isfinite(x) || return Inf
    r = x^(1 / e)
    while powd(r, e) < x
        r = nextfloat(r)
    end
    return r
end

"""
    gammau(ℓ, T) -> upper bound of γ_ℓ = ℓu/(1 − ℓu)

The constant of Higham's inner product bound `|fl(xᵀy) − xᵀy| ≤ γ_ℓ |x|ᵀ|y|` (no underflow) for a
real inner product of length `ℓ`, computed in any order, with or without fused multiply-add. For
complex data we use `γ_{ℓ+2}`: a complex product has relative error at most `√2 γ₂ ≤ γ₃`
(Higham, Lemma 3.5), and `(1 + γ₃)(1 + γ_{ℓ-1}) ≤ 1 + γ_{ℓ+2}`.
"""
function gammau(ℓ::Integer, ::Type{T} = Float64) where {T}
    ℓc = T <: Complex ? ℓ + 2 : ℓ
    u = unit_roundoff(real(T))
    nu = mulu(Float64(ℓc), u)
    nu < 0.5 || error("γ_$ℓc: ℓu ≥ 1/2, the rounding model does not apply")
    return divu(nu, subd(1.0, nu))
end
