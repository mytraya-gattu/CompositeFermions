# Incremental Elementary Symmetric Polynomial Update

## Setup

Given $N-1$ roots $\{r_1, \ldots, r_{N-1}\}$ and regularization coefficients $\{c_1, \ldots, c_b\}$, we define the **regularized elementary symmetric polynomials** $f_k$ via the generating function:

$$G(x) = \prod_{i=1}^{N-1} \left(1 + r_i \, c_k \, x\right) \quad \longleftrightarrow \quad f_k = \sum_{\substack{S \subseteq \{1,\ldots,N-1\} \\ |S|=k}} \left(\prod_{j=1}^{k} c_j\right) \prod_{i \in S} r_i$$

> [!NOTE]
> The $c_k$ factor couples to the *degree* $k$, not the root index. This makes the recurrence order-dependent — the regularization weights each order of the polynomial differently.

## Standard Recurrence (Full Computation)

The code computes $f_0, f_1, \ldots, f_b$ via the recurrence:

$$f_k^{(0)} = \delta_{k,0}$$

$$f_k^{(i)} = f_k^{(i-1)} + r_i \cdot c_k \cdot f_{k-1}^{(i-1)}, \qquad k = \min(i, b),\, \min(i,b)-1,\, \ldots,\, 1$$

where $f_k^{(i)}$ denotes the regularized symmetric polynomial of degree $k$ for the first $i$ roots. The final answer is $f_k = f_k^{(N-1)}$.

**Key point**: the loop over $k$ runs **backward** (high to low). This ensures that when computing $f_k^{(i)}$, the value $f_{k-1}^{(i-1)}$ on the RHS has not yet been overwritten — it is still the value from the previous root iteration.

**Cost**: $O(N \cdot b)$ per column.

## Incremental Update: Root Replacement

Suppose we have already computed $f_k = f_k^{(N-1)}$ for roots $\{r_1, \ldots, r_{N-1}\}$, and we want the result for the modified set where $r_m \to r_m'$ (a single root changes).

### Step 1: Remove the old root $r_m$ (forward recurrence)

From the recurrence $f_k^{(i)} = f_k^{(i-1)} + r_i \cdot c_k \cdot f_{k-1}^{(i-1)}$, we can solve for the "without root $r_m$" polynomials. However, we cannot directly peel off an arbitrary root $r_m$ from the middle — the recurrence built up roots in order $1, 2, \ldots, N-1$.

The crucial observation is that **elementary symmetric polynomials are symmetric in the roots**. The value $f_k^{(N-1)}$ does not depend on the order in which roots were processed. So we can always *pretend* that $r_m$ was the **last** root added, i.e., $f_k^{(N-1)} = f_k^{(N-2,\, \text{skip } m)} + r_m \cdot c_k \cdot f_{k-1}^{(N-2,\, \text{skip } m)}$.

Rearranging:

$$f_k^{(\text{without } r_m)} = f_k^{(\text{all})} - r_m \cdot c_k \cdot f_{k-1}^{(\text{without } r_m)}$$

This is a **forward recurrence** in $k = 1, 2, \ldots, b$, starting from $f_0^{(\text{without})} = 1$. At each step, $f_{k-1}^{(\text{without})}$ is already known from the previous iteration.

> [!IMPORTANT]
> The forward direction is essential here. We need $f_{k-1}^{(\text{without})}$ to compute $f_k^{(\text{without})}$, and the forward sweep guarantees this.

### Step 2: Add the new root $r_m'$ (backward recurrence)

Now we have $f_k^{(\text{without})}$ and want to include the new root $r_m'$:

$$f_k^{(\text{new})} = f_k^{(\text{without})} + r_m' \cdot c_k \cdot f_{k-1}^{(\text{without})}$$

This is a **backward recurrence** in $k = b, b-1, \ldots, 1$. We process high-to-low so that $f_{k-1}^{(\text{without})}$ on the RHS has not yet been overwritten by the new value.

> [!IMPORTANT]
> After Step 1, `dest` holds $f^{(\text{without})}$. Step 2 overwrites it with $f^{(\text{new})}$. The backward direction ensures we read $f_{k-1}^{(\text{without})}$ before writing $f_{k-1}^{(\text{new})}$.

### Combined Algorithm (in-place)

```julia
function update_symmetric_polynomials!(dest, r_old, r_new, b, reg_coeffs)
    # Step 1: Remove r_old (forward, k = 1 → b)
    for k in 1:b
        dest[k + 1] -= r_old * reg_coeffs[k] * dest[k]
    end
    # Step 2: Add r_new (backward, k = b → 1)
    for k in b:-1:1
        dest[k + 1] += r_new * reg_coeffs[k] * dest[k]
    end
end
```

**Cost**: $O(b)$ per column — independent of $N$.

## Application to Single-Particle MC Step

When particle `iter` moves in the Gibbs sampler, the `u_v_ratio_matrix` changes as follows:

- **Column $j \neq$ iter**: only the entry corresponding to particle `iter` changes (one root: $r_\text{old} \to r_\text{new}$). → Use **incremental update**, $O(b)$.
- **Column $j =$ iter**: all $N-1$ entries change (all roots are new). → Use **full recompute**, $O(Nb)$.

**Total cost per MC step**: $(N-1) \times O(b) + O(Nb) = O(Nb)$

compared to the previous $N \times O(Nb) = O(N^2 b)$.

## Worked Example: 4 Roots, $b = 3$

Take roots $\{a, b, c, d\}$ with regularization $\{c_1, c_2, c_3\}$. For clarity, set $c_k = 1$ (unregularized) first.

### Full computation (standard recurrence)

We process roots one at a time, sweeping $k$ from high to low:

| After root | $f_0$ | $f_1$ | $f_2$ | $f_3$ |
|-----------|-------|-------|-------|-------|
| init | $1$ | $0$ | $0$ | $0$ |
| $a$ | $1$ | $a$ | $0$ | $0$ |
| $b$ | $1$ | $a+b$ | $ab$ | $0$ |
| $c$ | $1$ | $a+b+c$ | $ab+ac+bc$ | $abc$ |
| $d$ | $1$ | $a+b+c+d$ | $ab+ac+ad+bc+bd+cd$ | $abc+abd+acd+bcd$ |

These are exactly $e_0, e_1, e_2, e_3$ of $\{a,b,c,d\}$. ✓

### Step 1: Remove root $d$ (forward, $k=1 \to 3$)

Starting from the final row $[1,\; e_1,\; e_2,\; e_3]$:

$$k=1: \quad f_1 \leftarrow f_1 - d \cdot f_0 = (a{+}b{+}c{+}d) - d = a{+}b{+}c$$

$$k=2: \quad f_2 \leftarrow f_2 - d \cdot f_1 = (ab{+}ac{+}ad{+}bc{+}bd{+}cd) - d(a{+}b{+}c) = ab{+}ac{+}bc$$

$$k=3: \quad f_3 \leftarrow f_3 - d \cdot f_2 = (abc{+}abd{+}acd{+}bcd) - d(ab{+}ac{+}bc) = abc$$

Result: $[1,\; a{+}b{+}c,\; ab{+}ac{+}bc,\; abc]$ — exactly $e_k(\{a,b,c\})$. ✓

> [!TIP]
> Notice why the **forward** direction is essential: at $k=2$, we needed the already-updated $f_1 = a{+}b{+}c$ (without $d$), not the original $f_1 = a{+}b{+}c{+}d$.

### Step 2: Add new root $d'$ (backward, $k=3 \to 1$)

Starting from $[1,\; a{+}b{+}c,\; ab{+}ac{+}bc,\; abc]$:

$$k=3: \quad f_3 \leftarrow f_3 + d' \cdot f_2 = abc + d'(ab{+}ac{+}bc) = abc{+}abd'{+}acd'{+}bcd'$$

$$k=2: \quad f_2 \leftarrow f_2 + d' \cdot f_1 = (ab{+}ac{+}bc) + d'(a{+}b{+}c) = ab{+}ac{+}ad'{+}bc{+}bd'{+}cd'$$

$$k=1: \quad f_1 \leftarrow f_1 + d' \cdot f_0 = (a{+}b{+}c) + d' = a{+}b{+}c{+}d'$$

Result: $[1,\; e_1(\{a,b,c,d'\}),\; e_2(\{a,b,c,d'\}),\; e_3(\{a,b,c,d'\})]$. ✓

> [!TIP]
> The **backward** direction is essential here: at $k=2$, we needed the still-untouched $f_1 = a{+}b{+}c$ (without $d'$), not the updated $f_1 = a{+}b{+}c{+}d'$.

### With regularization ($c_k \neq 1$)

Everything is identical, just multiply the root by $c_k$ at each step. The removal becomes:

$$f_k \leftarrow f_k - r_\text{old} \cdot c_k \cdot f_{k-1}$$

and addition:

$$f_k \leftarrow f_k + r_\text{new} \cdot c_k \cdot f_{k-1}$$

The directional constraints (forward for removal, backward for addition) are unchanged.

---

## Numerical Stability

Since the incremental update involves subtraction (Step 1), floating-point errors accumulate over many MC steps. To mitigate this:

- **Periodic full recompute**: every $N$ steps (i.e., once per full Gibbs sweep), do a full recomputation of all columns to reset accumulated error. This adds $O(N^2 b)$ cost per sweep but amortized over $N$ steps it's $O(Nb)$ per step — same as the incremental cost.
