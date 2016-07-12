from setuptools import setup, find_packages

setup(name="emme_modelling_tools",
      description="Collection of tools and modules for working with Emme-based travel demand forecasting models. " +
                  "Works best as a submodule of another project.",
      version="1.0",
      author="Peter Kucirek",
      author_email="pkucirek@pbworld.com",
      packages=find_packages(),
      requires=["numpy", "pandas", "shapely", "shapelib"],
      platforms="any"
      )
