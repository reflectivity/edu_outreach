# TODO

Comments from @arm61

- [x] Turn figure 2 into a tikz generated image (so that it matches the others)

Comments from Thomas Saerbeck

- [x] Explain focus on non-polarised reflectometry

Some comments from @andyfaff

- [x] Explanation of parameter uncertainty determination
- [x] How to use roughnesses **properly**
- [x] Roughness implementation
- [ ] Sensitivity limits (e.g. if film is 99% hydrated you might not see it?) -- (from @arm61 I am not really sure how best to fit this in, as the focus it on the mathematics)
- [x] Resolution
- [ ] Perhaps discussion of differential evolution is not necessary


Some comments from @timsnow

- [x] After eqn. 3 - is nuclear scattering length density or neutron SLD a more common turn of phrase? (You use SLD in figure 1's caption text, for example)
- [x] Figure 1's caption could do with clarifying to highlight that all three plots are representations of different aspects of the kinematic approach
- [x] Below eqn 5, remove exclamation mark in the body text, it's oddly out of place; yes it's impossible but it's not like the world ended
- [x] Whilst potentially beyond the scope of the article (but maybe only just) is it worth mentioning something about contrast measurements and co-refinement in global optimisation?
- [ ] Does Jos have a paper on the Bayesian / ML work they've been conducting? (For the end of section 4)


Some comments from Luke Clifton

- [x] One thing is I don’t understand how you go from equation 10 to equation 11. I maybe being thick but B = the product summation of M across all layer interfaces (Mn). Yet in equation 10 B1,2 / B1,1 = R(Q).

Some comments from @acaruana2009

- [ ] Should the appropriate handling of the roughness be expanded by giving some ideas of how to deal with large interfaces i.e. from interdiffusion, by slicing the interface? - [ ] It is good to make explicit that from too large roughness values not only the SLD you observe is unphysical, it also does not match the generated reflectivity simulation. This point is often missed. A few users I have spoken to think it is 'OK' to have a huge roughness because 'it gives the SLD profile they expect' missing the previous point.

- [x] Include original Storn and Price DE reference - Storn, R. Price, K. Journal of Global Optimization 1997, 11 (4), 341-359. DOI: 10.1023/A:1008202821328 
- [ ] Maybe include References for other fitting software that use DE (as long as they describe their usage of DE in the reference), to give the reader a broader selection of reading of which to draw from
- [ ] For the Bayesian section, I think some discussion about the number of samples required to reliably obtain the confidence interval is useful. i.e. 1e6 samples for the 95% interval. 
- [ ] Additionally, depending on the algorithm, the burn in can also include the convergence time, i.e. the chains of all of the population need to be in an equilibrium state - so they have all converged before taking samples 

