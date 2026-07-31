# Sample statistics & confidence interval — `sample_x.csv`

Sample size **n = 50** (50 numeric values under the `x` header).
Code that reproduces every number below is in [`sample_x_ci.py`](sample_x_ci.py)
(`python sample_x_ci.py`).

## (a) Sample mean

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i = \boxed{2.3589}$$

## (b) Sample variance

Using the **(n − 1)** divisor (the unbiased / course definition — the same
one R's `var()` and `numpy`'s `var(x, ddof=1)` use, **not** the population
`1/n` version):

$$S^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2 = \boxed{12.5031}$$

so the sample standard deviation is $S = \sqrt{S^2} = 3.5360$.

## (c) The `CI(x, alpha)` function

Because the population variance is unknown and estimated from the sample,
the interval uses the **t distribution** with `n − 1` degrees of freedom:

$$\bar{x} \pm t_{1-\alpha/2,\;n-1}\,\frac{S}{\sqrt{n}}$$

```python
import math
from scipy import stats

def CI(x, alpha):
    """(1 - alpha)*100% confidence interval for the mean of sample x."""
    n = len(x)
    xbar = sum(x) / n
    s2 = sum((xi - xbar) ** 2 for xi in x) / (n - 1)   # (n-1) divisor
    s = math.sqrt(s2)
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)          # two-sided critical value
    margin = t_crit * s / math.sqrt(n)
    return (xbar - margin, xbar + margin)
```

## (d) 95% confidence interval and interpretation

With `alpha = 0.05`, `t(0.975, df=49) = 2.0096`, margin of error = 1.0049:

$$\text{95\% CI} = (\,1.3540,\; 3.3638\,)$$

**Do we think the true mean is greater than 0?** **Yes.** The entire 95%
confidence interval lies above 0 (its lower endpoint, 1.35, is already
well above 0). A (1 − α) CI contains exactly the values of the mean that
would *not* be rejected by a two-sided test at level α, so 0 is rejected at
the 5% level. We are therefore 95% confident the true mean is positive —
in fact it is plausibly somewhere between about 1.35 and 3.36.
