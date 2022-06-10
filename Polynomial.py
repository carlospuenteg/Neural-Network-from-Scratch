class P:
    # Create the polynomial with the given coefficients
    def __init__(self, *coeffs):
        self.coeffs = coeffs

    # Get the value of the polynomial with a given x
    def p(self, x):
        return sum(coef*(x**(len(self.coeffs)-1-exp)) for exp,coef in enumerate(self.coeffs))

    # Show the polynomial
    def __str__(self):
        toret = ""
        max_exp = len(self.coeffs) - 1
        for exp,coef in enumerate(self.coeffs):
            exp = max_exp - exp
            if coef != 0:
                var = "x" if exp != 0 else ""
                sp = " " if exp != max_exp else ""
                coef = f"{sp}{'-' if coef < 0 else '+' if exp != max_exp else ''}{sp}{abs(coef) if abs(coef) != 1 or exp == 0 else ''}"
                exp = f"^{exp}" if exp > 1 else ""
                toret += f"{coef}{var}{exp}"
        return toret