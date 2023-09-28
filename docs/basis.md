# Bases

The Reduced-Basis Method seeks to reproduce a high-fidelity calculation to a
desired accuracy with a set of equations in a lower dimensional space. The main
step of the reduction is approximating the high-fidelity wave function by

$$
\phi_{\rm HF} \approx \hat{\phi} = \phi_0 + \sum_i c_i \tilde{\phi}_i~.
$$

These classes calculate and store the basis state $\phi_0$ and $\tilde{\phi}_i$.

::: rose.basis