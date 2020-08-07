# refl_maths

To build the refl_maths paper, the following is required: 

- anaconda or miniconda python
- [REVTex](https://journals.aps.org/revtex)

Then clone the repository and build using the following commands. 

```
cd refl_maths
conda env create --prefix ./refl_maths_env --file environment.yml
source activate ./refl_maths_env
snakemake
```